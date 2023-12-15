

def test(tokenizer, start, end, ds, device, model, max_length):
    print("testing...")

    for i in range(0, 100, 5):
        test_tokenized = tokenizer(start + ds[i]["source"] + end, return_tensors='pt')
        test_tokenized.to(device)
        result = model.generate(
            **test_tokenized, do_sample=True, top_k=5, max_length=max_length, top_p=0.95, num_return_sequences=3
        )
        print(i, "#" * 20)
        print("source:", ds[i]["source"])
        print("target:", ds[i]["target"])
        print("test result:", tokenizer.decode(result[0], skip_special_tokens=True))
        print("test result:", tokenizer.decode(result[1], skip_special_tokens=True))
        print("test result:", tokenizer.decode(result[2], skip_special_tokens=True))

