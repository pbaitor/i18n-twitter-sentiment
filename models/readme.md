Trained models for use with Tensorflow.JS

You can load each of them using code like:

```javascript

tf.loadLayersModel(`/models/anger/model.json`).then(model => /* do something with it */)

```

And get predictions from LASER sentence embeddings with:

```javascript

 model.predict(tf.tensor(embeddings)).data().then(prob => /* do something with the probability */)
 
 ```
 
 For more reference see https://www.tensorflow.org/js
