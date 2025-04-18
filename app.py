import process

def main():
    print("🧠 AI Chatbot Ready! Ask me anything based on the CSV context.")
    index, chunks = process.load_vector_store()

    while True:
        question = input("\n❓ Your question (or type 'exit'): ")
        if question.lower() == "exit":
            print("👋 Bye!")
            break

        answer = process.generate_answer(question, index, chunks)
        print("\n🤖 Answer:", answer)

if __name__ == "__main__":
    main()
