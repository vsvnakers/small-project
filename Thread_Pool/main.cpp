#include <iostream>
#include "ThreadPool.h"

void print_task(int id) {
    std::cout << "task " << id
              << " executed by thread "
              << std::this_thread::get_id()
              << std::endl;
}

int main() {
    ThreadPool pool(3);

    for (int i = 0; i < 10; ++i) {
        pool.submit([i] {
            print_task(i);
        });
    }

    pool.wait_all();
    std::cout << "main thread ends\n";
    return 0;
}
