/*
 * Context.hpp
 *
 *  Created on: Jun 7, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_CONTEXT_HPP_
#define AVOCADO_CORE_CONTEXT_HPP_

#include <Avocado/core/Device.hpp>
#include <Avocado/backend/backend_defs.h>

namespace avocado
{

	class Context
	{
		private:
			backend::avContextDescriptor_t m_data = backend::AVOCADO_INVALID_DESCRIPTOR;
			Device m_device;
		public:
			Context(Device device = Device::cpu());
			~Context();
			Context(const Context &other) = delete;
			Context(Context &&other);
			Context& operator=(const Context &other) = delete;
			Context& operator=(Context &&other);

			Device device() const noexcept;
			void synchronize() const;

			backend::avContextDescriptor_t getDescriptor() const noexcept;
			operator backend::avContextDescriptor_t() const noexcept;
	};

	backend::avContextDescriptor_t get_default_context(Device device) noexcept;

} /* namespace avocado */

#endif /* AVOCADO_CORE_CONTEXT_HPP_ */
