# Project-Lilium

[![license](https://img.shields.io/badge/license-GPL--3.0-red.svg)](https://github.com/juchengquan/project_lilium/blob/main/LICENSE)

A web framework easy to deploy large language model (LLM) applications.

## Usage

1. Download model weights from 🤗 [Huggingface Models](https://huggingface.co/models).
2. Make a copy from `./configs/models/codegen.yaml` and change the path of model weights.
3. Create the Python environment:

```bash
./scripts/create_env.sh -e your_py_env -v 3.10
```

4. Start the service by running:

```bash
./start.sh -e your_py_env -f /path/to/config.yaml
```

Or: 

```bash
export $PORT=8080 && \
source activate py3_10 && \
python ./main.py -f /path/to/config.yaml -p $PORT
```

## API

After starting the service, the API contract can be found at `$HOST:$PORT/docs`.

- Note that the default host is `LOCALHOST` and the default port is `8100`, which can be changed in `./start.sh`.

#### Normal Response

The normal response can be retrieved as shown below:

```python
import requests, json

url_infer = "http://127.0.0.1:8080/infer"

payload = json.dumps({
  "inputs": "What is the meaning of stonehenge?\n"
})
headers = {
  "Content-Type": "application/json"
}

response = requests.post(url_infer, headers=headers, data=payload)

if response.status_code == 200:
    print(response.json())
```

#### Streaming Response

The streaming response is supported as shown in the following example:

```python
import requests, json

url_infer_stream = "http://127.0.0.1:8100/infer_stream"

payload = json.dumps({
  "inputs": "What is the meaning of stonehenge?\n"
})
headers = {
  "Content-Type": "application/json"
}

with requests.post(url_infer_stream, data=payload, stream=True) as r:
    r.raise_for_status() # raise error if any
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True): # or, for line in r.iter_lines():
        res = json.loads(chunk)
        print(res["generated_text"])
```
