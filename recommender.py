import json
import re

class FashionRecommender:
    def __init__(self, model="llama3"):
        self.model_name  = model
        self.last_recommendation = "Detecting outfit..."
        self.cooldown    = 0
        self.cooldown_limit = 45
        self._ollama_ok  = True
        self._ollama     = None
        try:
            import ollama as _ol
            self._ollama = _ol
        except ImportError:
            print("ollama package not found – using rule-based recommendations.")
            self._ollama_ok = False

    def _clean_llm_output(self, raw_text):
        """Sanitize LLM output for clean, single-line display."""
        text = raw_text.strip()
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'^[-•]\s*', '', text)
        text = re.sub(r'"', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip('. ').strip()
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences:
            result = sentences[0].strip()
            if not result.endswith(('.', '!', '?')):
                result += '.'
            return result
        return text

    def generate_recommendation(self, outfit_data, gender="Unknown"):
        """Return a stylish fashion recommendation based on the outfit and gender."""
        upper = outfit_data.get("upper")
        lower = outfit_data.get("lower")

        if not upper and not lower:
            return "Detecting outfit..."

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.last_recommendation

        self.cooldown = self.cooldown_limit

        if self._ollama_ok and self._ollama is not None:
            items = []
            if upper: items.append(f"{upper['color']} {upper['label']}")
            if lower: items.append(f"{lower['color']} {lower['label']}")
            outfit_desc = ", ".join(items)

            prompt = (
                f"You are a premium fashion stylist. A {gender} is wearing: {outfit_desc}. "
                "Respond with ONLY one single sentence of styling advice. "
                "No bullet points, no lists, no markdown, no quotation marks. "
                "Just one clean, elegant sentence under 20 words."
            )
            try:
                response = self._ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a concise fashion advisor. Always reply in exactly one short sentence with no formatting."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw = response["message"]["content"]
                self.last_recommendation = self._clean_llm_output(raw)
                return self.last_recommendation
            except Exception as e:
                print(f"Ollama unavailable: falling back to premium style rules.")
                self._ollama_ok = False

        self.last_recommendation = self.get_premium_style_rules(upper, lower, gender)
        return self.last_recommendation

    def get_premium_style_rules(self, upper, lower, gender="Unknown"):
        """Stylish rule-based fallbacks."""
        g = gender.lower()
        
        items = []
        if upper: items.append(f"{upper['color']} {upper['label']}")
        if lower: items.append(f"{lower['color']} {lower['label']}")
        desc = " ".join(items).lower()

        if "dress" in desc:
            return "A classic dress shines with sleek heels and minimal, elegant jewelry."
        
        if g == "male":
            if "shirt" in desc:
                return "A crisp shirt tucked in with a leather belt creates a timeless, sharp profile."
            return "A clean, tailored fit is the key to elevated masculine style."
        
        if g == "female":
            if "top" in desc:
                return "Layering a refined top with delicate accessories creates a sophisticated, modern look."
            return "Focus on balanced proportions and elegant textures to elevate your silhouette."

        return "Experiment with different textures and fit to elevate your personal style."
