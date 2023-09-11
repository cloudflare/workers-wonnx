# webgpu sample using workerd

An example worker using [wonnx](https://github.com/webonnx/wonnx) for image classification on a webgpu-enabled version of [workerd](https://github.com/cloudflare/workerd).

## Upload model to local R2 simulator

This only needs to be done once.

    npx https://prerelease-registry.devprod.cloudflare.dev/workers-sdk/runs/6130596115/npm-package-wrangler-3919 r2 object put model-bucket-dev/opt-squeeze.onnx --local --file models/opt-squeeze.onnx

## Launch local development environment

    npx https://prerelease-registry.devprod.cloudflare.dev/workers-sdk/runs/6130596115/npm-package-wrangler-3919 dev

## Send request

    curl -v -F "file=@images/pelican.jpeg" http://localhost:8787
