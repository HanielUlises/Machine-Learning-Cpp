#define INTEGER int
#define BIGINTEGER long
#define REAL double

#define MIN_INTEGER std::numeric_limits<INTEGER>::min()
#define MAX_INTEGER std::numeric_limits<INTEGER>::max()
#define MIN_REAL -std::numeric_limits<REAL>::max()
#define MAX_REAL std::numeric_limits<REAL>::max()
#define MIN_POSITIVE_REAL std::numeric_limits<REAL>::min()
#define MIN_EPSILON std::numeric_limits<REAL>::epsilon()
#define NULL_INTEGER std::numeric_limits<INTEGER>::max()
#define NULL_REAL std::numeric_limits<REAL>::max()