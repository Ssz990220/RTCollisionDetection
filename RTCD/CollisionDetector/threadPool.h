#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <size_t NSTREAM>
class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();

    template <class F, class... Args>
    void enqueueTask(F&& f, Args&&... args);
    void waitForCompletion();

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    std::condition_variable completionCondition;
    std::atomic<bool> stop;
    std::atomic<int> tasksRemaining;

    void workerThread();
};

template <size_t NSTREAM>
ThreadPool<NSTREAM>::ThreadPool() : stop(false), tasksRemaining(0) {
    for (size_t i = 0; i < NSTREAM; ++i) {
        workers.emplace_back(&ThreadPool::workerThread, this);
    }
}

template <size_t NSTREAM>
ThreadPool<NSTREAM>::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template <size_t NSTREAM>
template <class F, class... Args>
void ThreadPool<NSTREAM>::enqueueTask(F&& f, Args&&... args) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace([f, args...]() { f(args...); });
        ++tasksRemaining;
    }
    condition.notify_one();
}

template <size_t NSTREAM>
void ThreadPool<NSTREAM>::workerThread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) {
                return;
            }
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (--tasksRemaining == 0) {
                completionCondition.notify_all();
            }
        }
    }
}

template <size_t NSTREAM>
void ThreadPool<NSTREAM>::waitForCompletion() {
    std::unique_lock<std::mutex> lock(queueMutex);
    completionCondition.wait(lock, [this] { return tasksRemaining == 0; });
}
