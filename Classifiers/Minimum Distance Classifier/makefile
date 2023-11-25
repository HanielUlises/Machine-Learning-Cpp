# Makefile for minimumDistanceClassifier project

# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++11 -Wall

# Source files
SRCS = main.cpp minimumDistanceClassifier.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)
# Executable name
TARGET = minimumDistanceClassifier

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
