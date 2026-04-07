import tf from '@tensorflow/tfjs';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede:
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })


    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(
                    `Epoch: ${epoch}: loss = ${log.loss}`
                )
            }
        }
    )

    return model
}


const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


const model = await trainModel(inputXs, outputYs)

