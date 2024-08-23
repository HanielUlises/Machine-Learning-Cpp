CXX = clang++
CXXFLAGS = -std=c++20 -Wall

TARGET = clustering_project.exe
SRCS = main.cpp KMeans.cpp ConfusionMatrix.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
