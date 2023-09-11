use image::io::Reader as ImageReader;
use image::{imageops::FilterType, Pixel};
use ndarray::{s, ArrayBase, Dim, OwnedRepr};
use std::convert::TryInto;
use std::{collections::HashMap, io::Cursor};
use wasm_bindgen::JsValue;
use wonnx::utils::TensorConversionError;
use worker::js_sys;
use worker::wasm_bindgen_futures;
use worker::worker_sys;

use worker::{durable_object, event, Context, Env, FormEntry, Method, Request, Response, Result};

use crate::squeeze_labels::LABELS;

mod squeeze_labels;

type ImageWeights = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>;

#[event(start)]
fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[durable_object]
pub struct Classifier {
    env: Env,
    session: Option<wonnx::Session>,
}

impl Classifier {
    async fn ensure_session(&mut self) -> Result<()> {
        match self.session {
            Some(_) => worker::console_log!("DO already has a session"),
            None => {
                // No session, so this should be the first request. In this case
                // we will fetch the model from R2, build a wonnx session, and
                // store it for subsequent requests.
                let model_bytes = fetch_model(&self.env).await?;
                worker::console_log!("Fetched model: {} bytes", model_bytes.len());
                let session = wonnx::Session::from_bytes(&model_bytes)
                    .await
                    .map_err(|err| err.to_string())?;
                worker::console_log!("session created in DO");
                self.session = Some(session);
            }
        };
        Ok(())
    }
}

#[durable_object]
impl DurableObject for Classifier {
    fn new(state: State, env: Env) -> Self {
        worker::console_log!("Building DO");
        Self { env, session: None }
    }

    async fn fetch(&mut self, mut req: Request) -> Result<Response> {
        worker::console_log!("Got DO request");

        // parse request and build model input data
        let request_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
            serde_json::from_str(&req.text().await?)?;
        let mut input_data = HashMap::new();
        input_data.insert("data".to_string(), request_data.as_slice().unwrap().into());

        self.ensure_session().await?;

        worker::console_log!("Start DO Compute");
        let result = self
            .session
            .as_ref()
            .unwrap() // we know the session exists
            .run(&input_data)
            .await
            .map_err(|err| err.to_string())?;
        worker::console_log!("Compute DO done");

        let probabilities: Vec<f32> = result
            .into_iter()
            .next()
            .ok_or("did not obtain a result tensor from session")?
            .1
            .try_into()
            .map_err(|err: TensorConversionError| err.to_string())?;

        let do_response = serde_json::to_string(&probabilities)?;
        Response::ok(do_response)
    }
}

// download model from R2
async fn fetch_model(env: &Env) -> Result<Vec<u8>> {
    let model_storage = env.bucket("MODEL_BUCKET")?;
    let model_file = env.var("MODEL_FILE")?.to_string();
    let obj = model_storage
        .get(model_file)
        .execute()
        .await?
        .ok_or("model file was not found")?;
    obj.body().ok_or("no model object body")?.bytes().await
}

#[event(fetch)]
pub async fn main(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    worker::console_log!("Starting request");

    if !matches!(req.method(), Method::Post) {
        return Response::error("Method Not Allowed", 405);
    }

    // obtain uploaded image
    let image_file: worker::File = match req.form_data().await?.get("file") {
        Some(FormEntry::File(buf)) => buf,
        Some(_) => return Response::error("`file` part of POST form must be a file", 400),
        None => return Response::error("missing `file`", 400),
    };
    let image_content = image_file.bytes().await?;
    let image = load_image(&image_content)?;

    // fetch durable object
    let namespace = env.durable_object("CLASSIFIER")?;
    let stub = namespace.id_from_name("A")?.get_stub()?;

    // obtain result probabilities and order by score
    let probabilities = execute_gpu_do(image, stub).await?;
    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();
    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // log a few probabilities
    for i in 0..3 {
        worker::console_log!(
            "Infered result: {} of class: {}",
            probabilities[i].1,
            LABELS[probabilities[i].0]
        );
    }

    // return only the most likely result
    Response::ok(LABELS[probabilities[0].0])
}

// build and send a request to the Classifier DO
async fn execute_gpu_do(image: ImageWeights, stub: worker::durable::Stub) -> Result<Vec<f32>> {
    let mut req_init = worker::RequestInit::new();
    req_init.method = Method::Post;
    let serialized = serde_json::to_string(&image)?;
    req_init.body = Some(JsValue::from_str(&serialized));
    let req = worker::Request::new_with_init("http://foo/test", &req_init)?;
    let mut do_resp = stub.fetch_with_request(req).await?;
    worker::console_log!("DO request done");

    let result: Vec<f32> = serde_json::from_str(&do_resp.text().await?)?;
    Ok(result)
}

// convert an image into weights
pub fn load_image(image_content: &Vec<u8>) -> Result<ImageWeights> {
    let image_buffer = ImageReader::new(Cursor::new(image_content))
        .with_guessed_format()?
        .decode()
        .map_err(|err| err.to_string())?
        .resize_to_fill(224, 224, FilterType::Nearest)
        .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    // Batch of 1
    Ok(array)
}
