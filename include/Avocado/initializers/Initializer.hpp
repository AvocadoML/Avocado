/*
 * Initializer.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INITIALIZERS_INITIALIZER_HPP_
#define AVOCADO_INITIALIZERS_INITIALIZER_HPP_

#include <memory>
#include <string>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Parameter;
}

namespace avocado
{
	class Initializer
	{
		public:
			Initializer() = default;
			Initializer(const Initializer &other) = delete;
			Initializer(Initializer &&other) = delete;
			Initializer& operator=(const Initializer &other) = delete;
			Initializer& operator=(Initializer &&other) = delete;
			virtual ~Initializer() = default;

			virtual void init(Parameter &param) = 0;

			virtual std::string name() const = 0;
			virtual Initializer* clone() const = 0;
			virtual Json serialize(SerializedObject &binary_data) const = 0;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data) = 0;
	};

	void registerInitializer(const Initializer &init);
	std::unique_ptr<Initializer> loadInitializer(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_INITIALIZERS_INITIALIZER_HPP_ */
