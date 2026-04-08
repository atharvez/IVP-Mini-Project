import ollama
import json

class FashionRecommender:
    def __init__(self, model="llama3"):
        self.model = model
        self.last_recommendation = "Detecting outfit..."
        self.cooldown = 0
        self.cooldown_limit = 30 # Only update recommendation every 30 frames to avoid flickering

    def generate_recommendation(self, colors):
        """
        Ask Ollama for a fashion recommendation based on detected colors.
        'colors' is a dictionary: {"Upper Wear": "Blue", "Lower Wear": "Black"}
        """
        if not colors:
            return "No clothing detected."

        # Manage cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return self.last_recommendation

        self.cooldown = self.cooldown_limit
        
        prompt = (
            f"I am wearing the following colors: {json.dumps(colors)}. "
            "Give me a very short (1 sentence) fashion recommendation or accessory idea. "
            "Be concise and stylish."
        )

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            self.last_recommendation = response['message']['content'].strip()
        except Exception as e:
            # Fallback simple rules if Ollama is not running
            print(f"Ollama Error: {e}")
            self.last_recommendation = self.get_rule_based_recommendation(colors)
            
        return self.last_recommendation

    def get_rule_based_recommendation(self, colors):
        upper = colors.get("Upper Wear", "").lower()
        lower = colors.get("Lower Wear", "").lower()
        
        if "blue" in upper and "black" in lower:
            return "Pair blue with white or beige for a brighter contrast."
        if "white" in upper:
            return "White goes with almost anything; try a bold accessory."
        if "red" in upper:
            return "Red makes a statement; balance it with neutral tones like gray or black."
        
        return "Experiment with contrasting textures to elevate your look!"
