/*
 * json.hpp
 *
 *  Created on: Oct 2, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_JSON_HPP_
#define AVOCADO_UTILS_JSON_HPP_

#include <string>
#include <variant>
#include <vector>
#include <stdexcept>

namespace avocado
{
	class Json;

	enum class JsonType
	{
		Null,
		Bool,
		Number,
		String,
		Array,
		Object
	};
	class Json
	{
		private:
			struct NullObject
			{
			};

			using json_null = NullObject;
			using json_bool = bool;
			using json_number = double;
			using json_string = std::string;
			using json_array = std::vector<Json>;

			using key_value_pair = std::pair<json_string, Json>;
			using json_object = std::vector<key_value_pair>;

			std::variant<json_null, json_bool, json_number, json_string, json_array, json_object> m_data;
		public:
			Json(JsonType type) noexcept;
			Json() noexcept;
			Json(bool b) noexcept;

			Json(int i) noexcept;
			Json(size_t i) noexcept;
			Json(float f) noexcept;
			Json(double d) noexcept;

			Json(const std::string &str);
			Json(const char *str);

			Json(const std::initializer_list<Json> &list);
			Json(const bool *list, size_t length);
			Json(const int *list, size_t length);
			Json(const double *list, size_t length);

			bool isNull() const noexcept;
			bool isBool() const noexcept;
			bool isNumber() const noexcept;
			bool isString() const noexcept;
			bool isArray() const noexcept;
			bool isArrayOfPrimitives() const noexcept;
			bool isObject() const noexcept;

			operator bool() const;
			operator int() const;
			operator size_t() const;
			operator float() const;
			operator double() const;
			operator std::string() const;

			bool getBool() const;
			int getInt() const;
			double getDouble() const;
			std::string getString() const;

			const Json& operator[](int idx) const;
			Json& operator[](int idx);

			const Json& operator[](const std::string &key) const;
			Json& operator[](const std::string &key);
			const Json& operator[](const char *key) const;
			Json& operator[](const char *key);

			const Json* find(const std::string &key) const noexcept;
			Json* find(const std::string &key) noexcept;
			bool hasKey(const std::string &key) const;
			const std::pair<std::string, Json>& entry(int idx) const;
			std::pair<std::string, Json>& entry(int idx);
			void append(const Json &other);

			// misc methods
			int size() const noexcept;
			void clear() noexcept;
			bool isEmpty() const noexcept;
			const char* storedType() const noexcept;

			// load/store
			std::string dump(int indent = -1) const;
			static Json load(const std::string &str);

		private:
			JsonType get_type() const noexcept;
			const json_array& as_array() const noexcept;
			json_array& as_array() noexcept;
			const json_object& as_object() const noexcept;
			json_object& as_object() noexcept;
	};

	class JsonKeyError: public std::logic_error
	{
		public:
			JsonKeyError(const char *method, const std::string &key);
	};
	class JsonTypeError: public std::logic_error
	{
		public:
			JsonTypeError(const char *method, const char *current_type);
	};
	class JsonParsingError: public std::logic_error
	{
		public:
			JsonParsingError(const char *method, const std::string &message);
	};

} /* namespace avocado */

#endif /* AVOCADO_UTILS_JSON_HPP_ */
