cmake_minimum_required(VERSION 3.20)
project(RealTimeCARFACSAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================
# Include directories
# ======================
# Eigen
include_directories("C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/eigen-3.4.0")

# PortAudio
include_directories("C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/portaudio/include")

# SDL2
include_directories("C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/SDL2/include")

# Matplot++
include_directories("C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/matplotplusplus/source")

# ======================
# Link directories / libraries
# ======================
link_directories(
    "C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/portaudio/lib"
    "C:/Users/maruk/Downloads/Python310/carfac-SAI/cpp/SDL2/lib/x64"
)

# ======================
# Sources
# ======================
set(SOURCES
    carfac.cc
    car.cc
    color.cc
    ear.cc
    image.cc
    pitchogram.cc
    pitchogram_pipeline.cc
    sai.cc
    realtime_sai.cc
    emscripten_bindings.cc
)

# ======================
# Define executable
# ======================
add_executable(${PROJECT_NAME} ${SOURCES})

# ======================
# Link libraries
# ======================
target_link_libraries(${PROJECT_NAME}
    portaudio_x64
    SDL2
)

# ======================
# Math constants for Windows
# ======================
add_definitions(-D_USE_MATH_DEFINES)

# ======================
# Optional: suppress some warnings
# ======================
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
