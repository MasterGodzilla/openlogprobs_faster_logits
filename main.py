from openlogprobs import extract_logprobs, ToyModel
toy_model = ToyModel(2, temperature = 1)
extracted_logprobs = extract_logprobs(toy_model, "i like pie", method="topk")
print (extract_logprobs)