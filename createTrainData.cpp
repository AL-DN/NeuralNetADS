#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

using namespace std;

int main()
{
    // Training Sets -- XOR

    cout << " topology: 2 4 1" << endl;
    for (unsigned i = 2000; i > 0; --i)
    {
        unsigned n1 = rand() % 2;
        unsigned n2 = rand() % 2;
        unsigned t = n1 ^ n2;
        assert(t == 1 || t == 0);
        cout << "in: " << n1 << ".0 " << n2 << ".0" << endl;
        cout << "out: " << t << ".0" << endl;
    }
}