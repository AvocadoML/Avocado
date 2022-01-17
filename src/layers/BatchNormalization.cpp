/*
 * BatchNormalization.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/BatchNormalization.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/math/batchnorm.hpp>
#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/static_block.hpp>
#include <Avocado/utils/testing_helpers.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(BatchNormalization());
	}

	BatchNormalization::BatchNormalization(const std::string &activation, bool useGamma, bool useBeta, int historySize) :
			Layer(std::string(activation))
	{
		m_use_gamma = useGamma;
		m_use_beta = useBeta;
		m_history_size = historySize;
	}

	BatchNormalization& BatchNormalization::useGamma(bool b) noexcept
	{
		m_use_gamma = b;
		return *this;
	}
	BatchNormalization& BatchNormalization::useBeta(bool b) noexcept
	{
		m_use_beta = b;
		return *this;
	}

	void BatchNormalization::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "BatchNormalization layer expects single input shape");
		m_running_mean = Tensor(Shape( { m_history_size, shapes[0].lastDim() }), dtype(), device());
		m_running_variance = Tensor(Shape( { m_history_size, shapes[0].lastDim() }), dtype(), device());
		m_input_shapes = shapes;
	}
	Shape BatchNormalization::getOutputShape() const
	{
		return getInputShape();
	}
	Shape BatchNormalization::getWeightShape() const
	{
		return Shape( { 2, getInputShape().lastDim() });
	}
	Shape BatchNormalization::getBiasShape() const
	{
		return Shape( { 2, getInputShape().lastDim() });
	}

	std::string BatchNormalization::name() const
	{
		return "BatchNormalization";
	}
	Json BatchNormalization::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_gamma"] = m_use_gamma;
		result["use_beta"] = m_use_beta;
		result["history_size"] = m_history_size;
		return result;
	}

	void BatchNormalization::changeContext(Context &context)
	{
		Layer::changeContext(context);
		m_running_mean.moveTo(device());
		m_running_variance.moveTo(device());
	}

	BatchNormalization* BatchNormalization::clone(const Json &config) const
	{
		return new BatchNormalization(config["nonlinearity"], config["use_gamma"], config["use_beta"], config["history_size"]); // @suppress("Ambiguous problem")
	}

	void BatchNormalization::init()
	{
		getWeights().getParam().setall(1);
		getBias().getParam().setall(0);
		m_running_mean.zeroall();
		m_running_variance.zeroall();
		m_total_steps = 0;
		m_running_id = -1;
	}
	void BatchNormalization::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		assert(same_device(context(), input[0], output));

		const int last_dim = getInputShape().lastDim();
		Tensor bias = getBias().getParam().view( { last_dim }, last_dim);
		Tensor scale = getWeights().getParam().view( { last_dim }, last_dim);

		if (input[0].shape().volumeWithoutLastDim() == 1)
		{
			Tensor estimatedMean = getBias().getParam().view( { last_dim });
			Tensor estimatedVariance = getWeights().getParam().view( { last_dim });
			math::batchNormInference(context(), 1, 0, input[0], output, scale, bias, estimatedMean, estimatedVariance, m_epsilon, m_nonlinearity);
		}
		else
		{
			m_running_id = (m_running_id + 1) % m_history_size;
			Tensor savedMean = m_running_mean.view( { last_dim }, m_running_id * last_dim);
			Tensor savedVariance = m_running_variance.view( { last_dim }, m_running_id * last_dim);

			math::batchNormForward(context(), 1, 0, input[0], output, scale, bias, savedMean, savedVariance, m_epsilon, m_nonlinearity);
			m_total_steps++;
		}
	}
	void BatchNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		assert(same_device(context(), input[0], output, gradient_prev[0], gradient_next));

		const int last_dim = getInputShape().lastDim();
		Tensor savedMean = m_running_mean.view( { last_dim }, m_running_id * last_dim);
		Tensor savedVariance = m_running_variance.view( { last_dim }, m_running_id * last_dim);
		Tensor scale = getWeights().getParam().view( { last_dim }, last_dim);

		Tensor biasUpdate = getBias().getUpdate().view( { last_dim }, last_dim);
		Tensor scaleUpdate = getWeights().getUpdate().view( { last_dim }, last_dim);

		math::batchNormBackward(context(), m_nonlinearity, 1, 0, input[0], output, gradient_prev[0], gradient_next, scale, savedMean, savedVariance,
				m_epsilon);
		math::batchNormUpdate(context(), 1, 1, input[0], gradient_next, scaleUpdate, biasUpdate, savedMean, savedVariance, m_epsilon);
	}
	void BatchNormalization::learn()
	{
		const int last_dim = getInputShape().lastDim();
		Tensor bias = getBias().getParam().view( { last_dim }, last_dim);
		Tensor scale = getWeights().getParam().view( { last_dim }, last_dim);

		if (!m_use_gamma)
			math::zeroTensor(context(), scale);
		if (!m_use_beta)
			math::zeroTensor(context(), bias);

		Layer::learn();

		if (!m_use_gamma)
			math::setTensor(context(), scale, 1);
		if (!m_use_beta)
			math::setTensor(context(), bias, 0);

		Tensor mean = getBias().getParam().view( { last_dim });
		Tensor variance = getWeights().getParam().view( { last_dim });

		const int tmp = std::min(m_history_size, m_total_steps);
		math::reduceTensor(context(), math::TensorReduceOp::AVG, 1, 0, m_running_mean.view( { tmp, last_dim }), mean);
		math::reduceTensor(context(), math::TensorReduceOp::AVG, 1, 0, m_running_variance.view( { tmp, last_dim }), variance);
	}

} /* namespace avocado */

