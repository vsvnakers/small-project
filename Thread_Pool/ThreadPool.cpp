#include "ThreadPool.h"

// 构造函数
ThreadPool::ThreadPool(size_t num_threads)
    : stop(false), active_tasks(0) {

    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            worker_loop();
        });
    }
}

// 析构函数
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
    }

    // 唤醒所有 worker
    cv_task.notify_all();

    // 等待线程结束
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// 提交任务
void ThreadPool::submit(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.push(std::move(task));
    }
    cv_task.notify_one();
}

// 等待所有任务完成
void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this] {
        return tasks.empty() && active_tasks == 0;
    });
}

// worker 线程主循环
void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mtx);

            cv_task.wait(lock, [this] {
                return stop || !tasks.empty();
            });

            if (stop && tasks.empty()) {
                return;
            }

            task = std::move(tasks.front());
            tasks.pop();
            ++active_tasks;
        }

        // 执行任务
        task();

        {
            std::unique_lock<std::mutex> lock(mtx);
            --active_tasks;
            if (tasks.empty() && active_tasks == 0) {
                cv_done.notify_all();
            }
        }
    }
}
