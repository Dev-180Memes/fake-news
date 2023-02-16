# fake-news

To run server install all dependencies in `requirements.txt` have to be installed. To install it first create a virtual environment to install the dependencies in.

```bash
cd fake-news

python -m venv env

source env/bin/activate
```

The commands above creates and activates a virtual environment where the dependencies can now be installed. To install the dependencies run the command `pip install -r requirements.txt`

After installing all dependencies you can go on and start the flask app

```bash
export FLASK_APP=server.py

export FLASK_APP=development

flask run
```

This should start the flask app and the prediction api should now be available at `http://127.0.0.1:5000/predict`. The api takes in json data containing the text input to be predicted.

```json
{
    'text': 'text input'
}
```

The server returns either `Real` or `Fake`
