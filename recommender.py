import json

class FashionRecommender:
    def __init__(self, model="llama3"):
        self.model_name  = model
        self.last_recommendation = "Detecting outfit..."
        self.cooldown    = 0
        self.cooldown_limit = 45   # frames between LLM calls
        self._ollama_ok  = True    # set False on first fatal error
        self._ollama     = None
        try:
            import ollama as _ol
            self._ollama = _ol
        except ImportError:
            print("ollama package not found – using rule-based recommendations.")
            self._ollama_ok = False

    def generate_recommendation(self, outfit):
        """
        Return a fashion recommendation string.
        Tries Ollama first; silently falls back to rule-based logic on any error.
        """
        if not outfit:
            return "No clothing detected."

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.last_recommendation

        self.cooldown = self.cooldown_limit

        if self._ollama_ok and self._ollama is not None:
            prompt = (
                f"I am wearing: {json.dumps(outfit)}. "
                "Give one short stylish fashion tip (1 sentence). Be concise."
            )
            try:
                response = self._ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                self.last_recommendation = response["message"]["content"].strip()
                return self.last_recommendation
            except Exception as e:
                print(f"Ollama unavailable ({type(e).__name__}): switching to rule-based mode.")
                self._ollama_ok = False   # don't retry this session

        self.last_recommendation = self.get_rule_based_recommendation(outfit)
        return self.last_recommendation

    def get_rule_based_recommendation(self, outfit):
        items = [f"{color} {item}" for item, color in outfit.items()]
        desc  = ", ".join(items).lower()

        if "dress" in desc:
            return "A classic dress shines with simple heels and minimal jewellery."
        if "coat" in desc or "sweater" in desc:
            return "Layer with a scarf in a complementary tone for a polished look."
        if "skirt" in desc:
            return "Pair a skirt with a tucked-in blouse for an effortless chic vibe."
        if "shirt" in desc and "t-shirt" not in desc:
            return "A semi-formal shirt looks great tucked in with a slim belt."
        if "t-shirt" in desc and "pants" in desc:
            return "Layer with a light jacket to elevate this casual street-wear look."
        if "t-shirt" in desc:
            return "A classic tee pairs perfectly with clean sneakers and a simple watch."
        if "red" in desc:
            return "Red is bold – keep accessories minimal and let the colour speak."
        if "blue" in desc:
            return "Blue tones pair beautifully with white or earth-toned accessories."
        if "black" in desc:
            return "An all-black outfit is timeless – add a metallic accent to pop."

        return "Experiment with layering different textures to elevate your outfit!"

