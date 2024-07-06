CXX = g++
CXXFLAGS = -std=c++11 -Wall

TARGET = clustering_project
SRCS = src/main.cpp src/KMeans.cpp src/ConfusionMatrix.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
    $(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $< -o $@

clean:
    rm -f $(OBJS) $(TARGET)

.PHONY: all clean
