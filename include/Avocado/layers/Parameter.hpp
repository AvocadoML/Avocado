/*
 * Parameter.hpp
 *
 *  Created on: Sep 29, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_PARAMETER_HPP_
#define AVOCADO_LAYERS_PARAMETER_HPP_

#include <Avocado/core/Tensor.hpp>
#include <Avocado/optimizers/Optimizer.hpp>
#include <Avocado/regularizers/Regularizer.hpp>
#include <Avocado/initializers/Initializer.hpp>

#include <memory>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Shape;
	class Device;
	class Context;
	enum class DataType;
}

namespace avocado
{
	class Parameter
	{
		private:
			Tensor m_param;
			std::unique_ptr<Tensor> m_update;
			std::unique_ptr<Optimizer> m_optimizer;
			std::unique_ptr<Regularizer> m_regularizer;
			std::unique_ptr<Initializer> m_initializer;
			int m_accumulated_updates = 0;
			bool m_is_trainable = true;

		public:
			Parameter(const Parameter &other);
			Parameter(Parameter &&other) = default;
			Parameter& operator=(const Parameter &other);
			Parameter& operator=(Parameter &&other) = default;

			Parameter(const Json &json, const SerializedObject &binary_data);
			Parameter(const Shape &shape, DataType dtype, Device device, bool trainable = true);

			void setTrainable(bool t);
			bool isTrainable() const noexcept;

			void setOptimizer(const Optimizer &optimizer) noexcept;
			Optimizer& getOptimizer() const;

			void setRegularizer(const Regularizer &regularizer) noexcept;
			Regularizer& getRegularizer() const;

			void setInitializer(const Initializer &initializer) noexcept;
			Initializer& getInitializer() const;

			Shape shape() const noexcept;
			DataType dtype() const noexcept;
			Device device() const noexcept;
			double getInvBatch() const noexcept;
			int getBatch() const noexcept;

			const Tensor& getParam() const;
			Tensor& getParam();
			Tensor& getUpdate();

			void moveTo(Device newDevice);
			void convertTo(const Context &context, DataType newType);
			void init(const Context &context);
			void learn(const Context &context);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_PARAMETER_HPP_ */
