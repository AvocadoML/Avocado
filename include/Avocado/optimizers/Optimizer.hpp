/*
 * Optimizer.hpp
 *
 *  Created on: Sep 29, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_OPTIMIZERS_OPTIMIZER_HPP_
#define AVOCADO_OPTIMIZERS_OPTIMIZER_HPP_

#include <memory>
#include <string>
#include <stdexcept>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Parameter;
	class Device;
	class Context;
}

namespace avocado
{

	class Optimizer
	{
		public:
			Optimizer() = default;
			Optimizer(const Optimizer &other) = delete;
			Optimizer(Optimizer &&other) = delete;
			Optimizer& operator=(const Optimizer &other) = delete;
			Optimizer& operator=(Optimizer &&other) = delete;
			virtual ~Optimizer() = default;

			virtual float getLearningRate() const noexcept = 0;
			virtual void setLearningRate(float lr) noexcept = 0;
			virtual int getSteps() const noexcept = 0;

			virtual void restart() noexcept = 0;
			virtual void moveTo(Device newDevice) = 0;
			virtual void learn(const Context &context, Parameter &param) = 0;

			virtual std::string name() const = 0;
			virtual Optimizer* clone() const = 0;
			virtual Json serialize(SerializedObject &binary_data) const = 0;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data) = 0;
	};

	void registerOptimizer(const Optimizer &opt);
	std::unique_ptr<Optimizer> loadOptimizer(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_OPTIMIZERS_OPTIMIZER_HPP_ */
