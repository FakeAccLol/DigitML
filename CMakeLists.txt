cmake_minimum_required(VERSION 3.15)
project(DigitML VERSION 0.1.0 LANGUAGES CXX)

# 1. Базовые настройки
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Опции проекта
option(BUILD_TESTING "Build unit tests" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# 4. Основной исполняемый файл
add_executable(${PROJECT_NAME} src/main.cpp src/dataset.cpp src/NN.cpp)

# 5. Конфигурация сборки
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR ENABLE_DEBUG)
    target_compile_options(${PROJECT_NAME}_lib 
        PRIVATE -g3 -O0 -Wall -Wextra -pedantic-errors
    )
    message(STATUS "Debug build configured")
endif()

add_custom_target(run 
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
)

add_custom_target(coverage
    COMMAND lcov --capture --directory . --output-file coverage.info
    COMMAND genhtml coverage.info --output-directory coverage_report
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
