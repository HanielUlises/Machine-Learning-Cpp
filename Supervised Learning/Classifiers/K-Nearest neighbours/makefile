# Makefile for k-nearest neighbours project

# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++17 -Wall

# Source files
SRCS = knn.cpp main.cpp 
# Object files
OBJDIR = objects
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
EXEC = KNN

all: $(OBJDIR) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(EXEC)
