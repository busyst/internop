use nalgebra::{DMatrix, DVector};
use rand::{rngs::ThreadRng, Rng};
use std::{f32::consts::PI, time::Instant};

#[derive(Clone, Copy)]
enum ActivationFunction {
    None,
    Sigmoid,
    TanH,
    ReLU,
    GeLU,
    SiLU,
}

struct DenseLayer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    activ: ActivationFunction,
    weights_momentum: DMatrix<f32>,
    bias_momentum: DVector<f32>,
}


impl DenseLayer {
    fn next_gaussian(mu: f32, sigma: f32, rng: &mut ThreadRng) -> f32 {
        let u1: f32 = 1.0 - rng.gen::<f32>(); // Uniform(0,1] random doubles
        let u2: f32 = 1.0 - rng.gen::<f32>();
        let rand_std_normal: f32 = ((-2.0 * u1.ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).sin(); // Box-Muller transform
        mu + sigma * rand_std_normal
    }
    fn new(input_size: usize, output_size: usize, activ: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0/(output_size as f32+input_size as f32)).sqrt(); // Example for small network, adjust accordingly

        let weights = DMatrix::<f32>::from_fn(output_size, input_size, |_, _| {
            DenseLayer::next_gaussian(0.0,std_dev, &mut rng)
        });
        let biases = DVector::<f32>::from_fn(output_size, |_, _| {
            DenseLayer::next_gaussian(0.0,std_dev, &mut rng)
        });

        let weights_momentum = DMatrix::<f32>::zeros(output_size, input_size);
        let bias_momentum = DVector::<f32>::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activ,
            weights_momentum,
            bias_momentum,
        }
    }

    fn feedforward(&self, input: &DVector<f32>) -> DVector<f32> {
        let z = &self.weights * input + &self.biases;
        match self.activ {
            ActivationFunction::None => z,
            ActivationFunction::Sigmoid => z.map(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::TanH => z.map(|x| x.tanh()),
            ActivationFunction::ReLU => z.map(|x| x.max(0.0)),
            ActivationFunction::GeLU => z.map(|x| 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * x + 0.044715 * x.powi(3)).tanh())),
            ActivationFunction::SiLU => z.map(|x| x / (1.0 + (-x).exp())),
        }
    }

    fn backprop(&mut self, input: &DVector<f32>, error: &DVector<f32>, learning_rate: f32, momentum: f32) -> DVector<f32> {
        let z = &self.weights * input + &self.biases;
        let dz = match self.activ {
            ActivationFunction::None => error.clone(),
            ActivationFunction::Sigmoid => {
                z.map(|x| (1.0 / (1.0 + (-x).exp())) * (1.0 - (1.0 / (1.0 + (-x).exp())))).component_mul(&error)
            },
            ActivationFunction::TanH => z.map(|x| 1.0 - x.tanh().powi(2)).component_mul(&error),
            ActivationFunction::ReLU => z.map(|x| if x > 0.0 { 1.0 } else { 0.0 }).component_mul(&error),
            ActivationFunction::GeLU => z.map(|x|   0.5+
                                                        (0.398942*x+0.0670725*x*x*x)/((0.797885*x+0.044715*x*x*x).cosh().powi(2))+
                                                        (0.5*(0.797885*x+0.044715*x*x*x).sinh())/((0.797885*x+0.044715*x*x*x).cosh())
                                                        ).component_mul(&error),
            ActivationFunction::SiLU => z.map(|x| (x.exp()*(x+x.exp()+1.0))/((x.exp()+1.0).powi(2))).component_mul(&error),
        };

        self.weights_momentum = momentum * &self.weights_momentum + learning_rate * (dz.clone() * input.transpose()) * (1.0 - momentum);
        self.bias_momentum = momentum * &self.bias_momentum + learning_rate * dz.clone() * (1.0 - momentum);

        self.weights += &self.weights_momentum;
        self.biases += &self.bias_momentum;

        self.weights.transpose() * dz
    }
}

struct NeuralNetwork {
    layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize], default_activ: ActivationFunction) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1], default_activ));
        }
        NeuralNetwork { layers }
    }

    fn feedforward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.layers.iter().fold(input.clone(), |acc, layer| layer.feedforward(&acc))
    }

    fn backprop(&mut self, inputs: &[DVector<f32>], error: &DVector<f32>, learning_rate: f32, momentum: f32) {
        let mut error = error.clone();
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            // Use the stored input from the forward pass
            let input = &inputs[i];
            error = layer.backprop(input, &error, learning_rate, momentum);
        }
    }
}
fn main() {
    let input_data = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];

    let expected_outputs = vec![
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![0.0]),
    ];

    let mut network = NeuralNetwork::new(&[2, 5, 1], ActivationFunction::None);

    let learning_rate = 0.01;
    let momentum = 0.75;
    let epochs = 10000;
    let now = Instant::now();
    for i in 0..epochs {
        let mut total_error = 0.0;
        for (input, expected) in input_data.iter().zip(expected_outputs.iter()) {
            let output = network.feedforward(input);
            let error = expected - &output;
            total_error += error.iter().map(|&x| x.powi(2)).sum::<f32>();

            let mut layer_inputs = vec![input.clone()];
            for layer in &mut network.layers {
                let output = layer.feedforward(layer_inputs.last().unwrap());
                layer_inputs.push(output);
            }
            network.backprop(&layer_inputs, &error, learning_rate, momentum);
        }

        if i % 1000 == 0 {
            println!("Error at step {}: {}", i, total_error);
        }
    }
    println!("Training time:{}", now.elapsed().as_millis());
    // Testing the trained network
    println!("Testing trained XOR network:");
    for input in input_data {
        let output = network.feedforward(&input);
        println!("Input: {:?}, Output: {:?}", input, output);
    }
}