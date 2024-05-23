# Makefile for logistic regression project

# Compiler
CXX = clang++
# Compiler flags
CXXFLAGS = -std=c++17 -Wall

# Source files
SRCS = logistic_regression.cpp main.cpp 
# Object files
OBJDIR = objects
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
EXEC = logistic_regression.exe

all: $(OBJDIR) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(EXEC)
