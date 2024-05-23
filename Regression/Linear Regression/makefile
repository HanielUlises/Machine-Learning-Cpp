CXX = g++
CXXFLAGS = -std=c++17 -I/usr/include/python3.10 -I/home/haniel/Libraries -I/home/haniel/.local/lib/python3.10/site-packages/numpy/core/include
LDFLAGS = -lpython3.10

OBJS = main.o linear_regression.o 

lr: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) lr
