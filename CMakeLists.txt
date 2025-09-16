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

