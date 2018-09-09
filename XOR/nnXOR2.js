//training data
const xs = tf.tensor2d([
  [0,0],[0,1],[1,0],[1,1]
]);

console.log(xs.print());

const ys = tf.tensor1d([
  0,1,1,0
]);

console.log(ys.print());

async function learnLinear(){
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape:[2],
    activation: 'relu',
    units: 8
  }));

  model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 1
  }));

  //parameters for compile
  const learningRate = 0.1;
  const optimizer = tf.train.sgd(learningRate); //stochastic gradient descent

  //Before training a model, you need to configure the learning process, which is done via the compile method.
  model.compile({
    optimizer: optimizer,
    loss: tf.losses.meanSquaredError,
    metrics: ['binaryAccuracy'],
  });

  const response = await model.fit(xs, ys, {
    epochs: 1000
  });
  console.log("loss:", response.history.loss[0])
  const prediction = model.predict(xs);
  console.log(prediction.print());
};

learnLinear();
