package main

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

const AppName = "Go Introduction demo"
const MaxWorkser = 3

var version = "1.0"

type User struct {
	ID   int
	Name string
	Age  int
}

func (u User) SayHello() string {
	return fmt.Sprintf("Hi, I'm %s, age is %d. ", u.Name, u.Age)
}

func (u *User) Brithday() {
	u.Age++
}

var ErrInvalidAge = errors.New("Age can't be negative.")

func NewUser(id int, name string, age int) (*User, error) {
	if age < 0 {
		return nil, ErrInvalidAge
	}
	return &User{ID: id, Name: name, Age: age}, nil
}

// ProcessUser simulates processing a user in a separate goroutine.
// This function runs concurrently with other goroutines.
func ProcessUser(user *User, resultChan chan<- string, wg *sync.WaitGroup) {
	// defer wg.Done() ensures that when this function finishes (normally or via panic),
	// it tells the WaitGroup: "one task is done".
	defer wg.Done()

	fmt.Printf("Start to handle user: %s (goroutine ID simulated)\n", user.Name)

	// Simulate time-consuming work (e.g., network request, DB query).
	time.Sleep(time.Millisecond * 500)

	greeting := user.SayHello()

	// Send the result to the channel. This is safe and avoids shared memory.
	resultChan <- greeting
}

func main() {
	fmt.Printf("Activated %s  v%s  ...", AppName, version)
	fmt.Printf("==================")

	var language string = "Go"
	creator := "Google"
	var year int
	year = 2009

	fmt.Printf("Language: %s, Creator: %s, Year: %d\n .", language, creator, year)

	userMap := make(map[int]*User)

	users := []struct {
		id   int
		name string
		age  int
	}{
		{1, "Jim", 25},
		{2, "Keven", -5},
		{3, "Belly", 30},
	}

	for _, u := range users {
		user, err := NewUser(u.id, u.name, u.age)
		if err != nil {
			fmt.Printf("❌ created user %s failed: %v\n", u.name, err)
			continue
		}
		userMap[user.ID] = user
		fmt.Printf("✅ created user: %s \n", user.Name)
	}

	if user, exists := userMap[1]; exists {
		fmt.Printf("\n before: %s\n", user.SayHello())
		user.Brithday()
		fmt.Printf("after birthday: %s\n", user.SayHello())
	}

	fmt.Printf("\n start goroutinue dealing users...")

	// Create a buffered channel to collect results from goroutines.
	// Buffer size = number of users, so sending won't block immediately.
	resultChan := make(chan string, len(userMap))

	// WaitGroup is used to wait for all goroutines to finish.
	var wg sync.WaitGroup

	// Launch a goroutine for each user.
	for _, user := range userMap {
		wg.Add(1) // Tell WaitGroup: "we are starting one more task"

		// Start concurrent execution with 'go'.
		go ProcessUser(user, resultChan, &wg)
	}

	// Start a dedicated goroutine to close the channel after all workers finish.
	go func() {
		wg.Wait()         // Block until all tasks call wg.Done()
		close(resultChan) // Close channel to signal "no more data"
	}()

	// Main goroutine receives results from the channel.
	// 'for range' automatically stops when the channel is closed.
	fmt.Println("\n Receive result: ")
	for result := range resultChan {
		fmt.Println(" -> ", result)
	}
	fmt.Println("All jobs completed ...")
}
