//============================================================================
// Name        : Avocado.cpp
// Author      : Maciej Kozarzewski
//============================================================================

#include <iostream>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/math/tensor_operations.hpp>

#include <Avocado/graph/Graph.hpp>
#include <Avocado/layers/Dense.hpp>
#include <Avocado/layers/Flatten.hpp>
#include <Avocado/layers/Softmax.hpp>
#include <Avocado/layers/Activation.hpp>
#include <Avocado/optimizers/SGD.hpp>
#include <Avocado/optimizers/ADAM.hpp>
#include <Avocado/losses/MeanSquareLoss.hpp>
#include <Avocado/losses/CrossEntropyLoss.hpp>
#include <Avocado/utils/file_helpers.hpp>

#include <Avocado/expression/Expression.hpp>
#include <Avocado/expression/autograd.hpp>

#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>

using namespace avocado;
using namespace avocado::backend;

class MNIST
{
	private:
		const std::string path = "/home/maciek/Downloads/mnist/";
		mutable std::vector<int> train_ordering;
		mutable std::vector<int> test_ordering;
		mutable size_t train_index = 0;
		mutable size_t test_index = 0;
	public:
		Tensor train_images;
		Tensor test_images;
		Tensor train_labels;
		Tensor test_labels;
		MNIST()
		{
			train_images = load_images(path + "train-images-idx3-ubyte", 60000);
			train_labels = load_labels(path + "train-labels-idx1-ubyte", 60000);
			train_ordering.assign(train_images.dimension(0), 0);
			std::iota(train_ordering.begin(), train_ordering.end(), 0);

			test_images = load_images(path + "t10k-images-idx3-ubyte", 10000);
			test_labels = load_labels(path + "t10k-labels-idx1-ubyte", 10000);
			test_ordering.assign(test_images.dimension(0), 0);
			std::iota(test_ordering.begin(), test_ordering.end(), 0);
		}
		void printSample(int index) const
		{
			std::cout << "label = " << train_labels.get<int>( { index }) << '\n';
			std::cout << "┌────────────────────────────────────────────────────────┐\n";
			for (int i = 0; i < 28; i++)
			{
				std::cout << "│";
				for (int j = 0; j < 28; j++)
				{
					if (train_images.get<float>( { index, i * 28 + j }) < 0.2f)
						std::cout << "  ";
					else
					{
						if (train_images.get<float>( { index, i * 28 + j }) < 0.4f)
							std::cout << "░░";
						else
						{
							if (train_images.get<float>( { index, i * 28 + j }) < 0.6f)
								std::cout << "▒▒";
							else
							{
								if (train_images.get<float>( { index, i * 28 + j }) < 0.8f)
									std::cout << "▓▓";
								else
									std::cout << "██";
							}
						}
					}
				}
				std::cout << "│\n";
			}
			std::cout << "└────────────────────────────────────────────────────────┘\n\n";
		}
		void packTrainSamples(Tensor &input, Tensor &target)
		{
			pack_samples(input, target, train_images, train_labels, train_ordering, train_index);
		}
		void packTestSamples(Tensor &input, Tensor &target)
		{
			pack_samples(input, target, test_images, test_labels, test_ordering, test_index);
		}
	private:
		void pack_samples(Tensor &input, Tensor &target, const Tensor &images, const Tensor &labels, std::vector<int> &ordering, size_t &index) const
		{
			assert(input.firstDim() == target.firstDim());
			std::vector<float> host_input(input.volume(), 0.0f);
			std::vector<float> host_target(target.volume(), 0.0f);
			for (int i = 0; i < input.firstDim(); i++)
			{
				if (index >= ordering.size())
				{
					std::random_shuffle(ordering.begin(), ordering.end());
					index = 0;
				}
				const int sample_index = ordering.at(index) % 1000;
				index++;

				Tensor tmp = const_cast<Tensor&>(images).view(Shape( { 28, 28 }), sample_index * 28 * 28);
				tmp.copyToHost(host_input.data() + i * 28 * 28, 28 * 28);
				const int label = labels.get<int>( { sample_index });
				host_target.at(i * 10 + label) = 1.0f;
			}

			Tensor tmp_input(input.shape(), input.dtype(), Device::cpu());
			Tensor tmp_target(target.shape(), target.dtype(), Device::cpu());
			input.copyFromHost(host_input.data(), input.volume());
			target.copyFromHost(host_target.data(), target.volume());
		}
		Tensor load_images(const std::string &path, int n)
		{
			std::fstream stream(path, std::fstream::in);
			std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(n * 28 * 28);
			stream.read(reinterpret_cast<char*>(buffer.get()), 16); // skip header
			stream.read(reinterpret_cast<char*>(buffer.get()), n * 28 * 28);
			Tensor images( { n, 28 * 28 }, "float32", Device::cpu());
			for (int i = 0; i < n; i++)
				for (int j = 0; j < 28 * 28; j++)
					images.set<float>(static_cast<float>(buffer[i * 28 * 28 + j]) / 255.0f, { i, j });
			return images;
		}
		Tensor load_labels(const std::string &path, int n)
		{
			std::fstream stream(path, std::fstream::in);
			std::unique_ptr<char[]> buffer = std::make_unique<char[]>(n);
			stream.read(buffer.get(), 8); // skip header

			Tensor labels( { n }, "int32", Device::cpu());
			stream.read(buffer.get(), n);
			for (int i = 0; i < n; i++)
				labels.set<int>(buffer[i], { i });
			return labels;
		}
};

double get_accuracy(const Tensor &output, const Tensor &target)
{
	assert(output.firstDim() == target.firstDim());
	double correct = 0;
	for (int i = 0; i < output.firstDim(); i++)
	{
		int output_idx = 0;
		float output_max = output.get<float>( { i, 0 });
		for (int j = 0; j < output.lastDim(); j++)
			if (output.get<float>( { i, j }) > output_max)
			{
				output_max = output.get<float>( { i, j });
				output_idx = j;
			}

		int target_idx = 0;
		for (int j = 0; j < target.lastDim(); j++)
			if (target.get<float>( { i, j }) == 1.0f)
				target_idx = j;

		correct += static_cast<int>(target_idx == output_idx);
	}
	return correct;
}

void train_mnist()
{
	Device::cpu().setNumberOfThreads(1);
	MNIST dataset;
//	for (int i = 0; i < 10; i++)
//		dataset.printSample(i);

	int batch_size = 2;
	Graph model;
	auto x = model.addInput( { batch_size, 28 * 28 });
//	x = model.add(Flatten(), x);
//	x = model.add(Dense(30, "relu"), x);
	x = model.add(Dense(10, "sigmoid"), x);
//	x = model.add(Activation("sigmoid"), x);
	model.addOutput(x, MeanSquareLoss());
	model.init();
	model.setOptimizer(ADAM(1.0e-3));

//	auto x = model.addInput( { batch_size, 28, 28, 1 });
//	x = model.add(Conv2D(32, 5, "linear"), x);
//	x = model.add(BatchNormalization("relu"), x);
//	x = model.add(Conv2D(16, 3, "linear"), x);
//	x = model.add(BatchNormalization("relu"), x);
//	x = model.add(Flatten(), x);
//	x = model.add(Dense(30, "linear"), x);
//	x = model.add(BatchNormalization("relu"), x);
//	x = model.add(Dense(10, "linear"), x);
//	x = model.add(Softmax(), x);
//	model.addOutput(x, KLDivergenceLoss());
//	model.init();
//	model.setOptimizer(ADAM(1.0e-5f));
//	model.setRegularizer(RegularizerL2(1.0e-4f));
//	model.moveTo(Device::cuda(0));

	for (int i = 0; i < 1; i++)
	{
		double avg_loss = 0.0;
		double avg_acc = 0.0;
		int counter = 0;
		for (int j = 0; j < 5; j += batch_size, counter++)
		{
			dataset.packTrainSamples(model.getInput(), model.getTarget());
			model.forward(batch_size);
			model.backward(batch_size);
			auto loss = model.getLoss(batch_size);
			avg_loss += loss.at(0).get<float>();
			avg_acc += get_accuracy(model.getOutput(), model.getTarget());

			model.learn();
		}
		for (int k = 0; k < 10; k++)
			std::cout << k << " " << model.getOutput().get<float>( { 0, k }) << " " << model.getTarget().get<float>( { 0, k }) << '\n';
		std::cout << '\n';
		std::cout << "Epoch " << i << " : train loss = " << avg_loss / (counter * batch_size) << ", acc = " << avg_acc / (counter * batch_size)
				<< '\n';

//		avg_loss = 0.0;
//		avg_acc = 0.0;
//		counter = 0;
//		for (int j = 0; j < 1; j += batch_size, counter++)
//		{
//			dataset.packTestSamples(model.getInput(), model.getTarget());
//			model.forward(batch_size);
//			auto loss = model.getLoss(batch_size);
//			avg_loss += loss.at(0).get<float>();
//			avg_acc += get_accuracy(model.getOutput(), model.getTarget());
//		}
//		std::cout << "Epoch " << i << " : test loss  = " << avg_loss / counter << ", acc = " << avg_acc / (counter * batch_size) << '\n' << '\n';
	}
//	SerializedObject so;
//	Json json = model.save(so);
//	FileSaver fs("mnist_network.bin");
//	fs.save(json, so);
}

int main()
{
	Expression e;
	auto x = e.input( { 1 });
	auto w = e.input( { 1 });
	auto z = e.sigmoid(x * w);

	e.output(z);
	auto target = e.target(z);
	e.loss(e.constant(0.5) * e.square(z - target));
//	std::cout << e.toString() << '\n';
//	e.sort();

	std::cout << e.toString() << '\n';
	Expression b = e.getBackprop();
	std::cout << '\n' << b.toString() << '\n';

//	auto x = e.input( { 2, 784 });
//	auto w = e.input( { 10, 784 });
//	auto z = e.sigmoid(e.matmul(x, w, 'n', 't'));
//
//	auto target = e.output(z);
//	auto loss = e.reduce_add(e.constant(0.5) * e.square(z - target));
//	e.loss(loss);

//	auto z = e.input();
//	auto t = e.mul(x, y) + z;

//	Expression other = e.clone();
//	other.invert();

//	Expression backprop = Autograd::getBackward(e);
//	std::cout << e.toString();
//	std::cout << '\n' << other.toString();
//	e.invert();
//	std::cout << e.toString();

//	std::cout << Device::hardwareInfo() << std::endl;

//	train_mnist();
	std::cout << "END" << std::endl;
	return 0;

	DataType dtype = DataType::FLOAT16;
	Scalar zero = Scalar::zero(dtype);
	Scalar one = Scalar::one(dtype);

	std::cout << zero.toString() << "\n" << one.toString() << '\n';

	Context context(Device::cuda(0));
	Tensor t1( { 100 }, DataType::FLOAT16, Device::cuda(0));
	Tensor t2( { 100 }, DataType::FLOAT32, Device::cuda(0));
//	initForTest(t1, 0.0f);
//	initForTest(t2, 0.0f);

	Tensor t3(t1.shape(), DataType::FLOAT32, t1.device());
	math::changeType(context, t3, t1);

//	std::cout << diffForTest(t2, t3) << '\n';

//	Context context(Device::cpu());
//	context.synchronize();
//
//	Tensor t1(Shape( { 10, 10, 10 }), DataType::FLOAT32, Device::cpu());
//	t1.setall(1.1f);
//	t1.set(2.5f, { 1, 2, 3 });
//
//	Tensor t2(Shape( { 10, 10, 10 }), DataType::INT32, Device::cpu());
//
//	math::changeType(context, t2, t1);
//	context.synchronize();
//
//	std::cout << t1.get<double>( { 1, 2, 3 }) << ' ' << t1.get<float>( { 1, 2, 2 }) << '\n';
//	std::cout << t2.get<int>( { 1, 2, 3 }) << ' ' << t2.get<int>( { 1, 2, 2 }) << '\n';

	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
	return 0;
}
