#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool {
public:
    // 构造 / 析构
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // 提交任务
    void submit(std::function<void()> task);

    // 等待所有任务完成
    void wait_all();

    // 禁止拷贝
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    // worker 主循环
    void worker_loop();

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mtx;
    std::condition_variable cv_task;
    std::condition_variable cv_done;

    bool stop;
    size_t active_tasks;
};

#endif // THREAD_POOL_H
