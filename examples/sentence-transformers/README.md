A simple example of exporting a [transformer](https://huggingface.co/docs/transformers/index) model with Python, then loading it into tract to make predictions.

# To Use

First export the pre-trained transformer model using Python and PyTorch, then build the wasm module with wasm-pack, and finally serve the example with a simple http server.

``` shell
python export.py
wasm-pack build --target web --profiling --no-typescript
python -m http.server
```

Finally, open http://localhost:8000 in your browser.
