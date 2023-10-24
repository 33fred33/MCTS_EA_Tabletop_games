import multiprocessing

def write_to_file(filename, data, lock):
    with lock:
        with open(filename, 'a') as file:
            file.write(data)

if __name__ == "__main__":
    # Create a lock for synchronization
    lock = multiprocessing.Lock()
    print("Number of cpu : ", multiprocessing.cpu_count())
    
    # Define the filenames and data for each process
    filenames = ["file1.txt", "file2.txt"]
    data = ["Process 1 data", "Process 2 data"]
    
    # Create and start processes
    processes = []
    for i in range(len(filenames)):
        p = multiprocessing.Process(target=write_to_file, args=(filenames[i], data[i], lock))
        processes.append(p)
        p.start()
    
    # Wait for processes to finish
    for p in processes:
        p.join()