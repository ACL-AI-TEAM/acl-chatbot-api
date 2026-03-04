import logging
import re
from typing import List, Dict, Optional, Tuple
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "fr": """Tu es l'assistant intelligent d'AfricTivistes CitizenLab (ACL), une organisation panafricaine dédiée à la promotion de la citoyenneté numérique et de la démocratie en Afrique.

Ton rôle est d'aider les utilisateurs à trouver des informations sur :
- Les programmes et activités d'AfricTivistes CitizenLab
- La citoyenneté numérique en Afrique
- Les acteurs du CiviTech africain
- Les rapports et études sur l'internet citoyen en Afrique

Instructions STRICTES :
1. Réponds TOUJOURS en français
2. Utilise les extraits de contexte fournis pour répondre avec précision
3. Si l'information n'est pas dans le contexte, dis-le honnêtement
4. Sois concis, clair et professionnel
5. Ne fabrique jamais d'informations
6. INTERDIT ABSOLU : Ne mentionne JAMAIS de fichiers, CSV, sources, références
7. INTERDIT ABSOLU : N'écris JAMAIS de section Sources, Références, 📚 ou similaire
8. INTERDIT ABSOLU : N'utilise JAMAIS d'astérisques ou markdown
9. Réponds UNIQUEMENT avec du texte brut en paragraphes directs
10. Termine ta réponse dès que tu as répondu à la question — rien d'autre après

Contexte :
{context}""",

    "en": """You are the intelligent assistant of AfricTivistes CitizenLab (ACL), a pan-African organization dedicated to promoting digital citizenship and democracy in Africa.

STRICT Instructions:
1. ALWAYS respond in English
2. Use the context extracts provided to answer accurately
3. If information is not in the context, say so honestly
4. Be concise, clear and professional
5. Never fabricate information
6. ABSOLUTE FORBIDDEN: Never mention files, CSV, sources or references
7. ABSOLUTE FORBIDDEN: Never write a Sources, References or 📚 section
8. ABSOLUTE FORBIDDEN: Never use asterisks or markdown
9. Respond ONLY with plain text in direct paragraphs
10. End your response as soon as you have answered — nothing else after

Context:
{context}"""
}


def clean_response(text: str) -> str:
    """Nettoyage agressif et garanti de la réponse"""

    # Supprime TOUT après Sources / Références / 📚 (capture tout jusqu'à la fin)
    text = re.sub(
        r'[\n\r]*(?:📚\s*)?(?:Sources?|Références?|References?)\s*:[\s\S]*$',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Supprime chaque ligne contenant .csv / .pdf / .txt
    text = re.sub(
        r'^[^\n]*\.(csv|pdf|txt)[^\n]*$',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Supprime chaque ligne contenant (XX%)
    text = re.sub(
        r'^[^\n]*\(\d+%\)[^\n]*$',
        '',
        text,
        flags=re.MULTILINE
    )

    # Supprime 📚 et tout ce qui suit sur la ligne
    text = re.sub(r'📚[^\n]*', '', text)

    # Supprime **texte** et *texte*
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text, flags=re.DOTALL)

    # Supprime les titres markdown
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Supprime les listes markdown
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)

    # Supprime les lignes vides multiples
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


class GroqService:
    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if not settings.GROQ_API_KEY:
            logger.error("❌ GROQ_API_KEY manquante!")
            return
        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info(f"✅ Groq initialisé (modèle: {settings.GROQ_MODEL})")
        except Exception as e:
            logger.error(f"❌ Erreur Groq: {e}")

    def generate_response(
        self,
        user_message: str,
        context: str,
        history: List[Dict],
        language: str = "fr"
    ) -> Tuple[str, Optional[int]]:

        if not self.client:
            return "❌ Service LLM non disponible. Vérifiez GROQ_API_KEY.", None

        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["fr"]).format(context=context)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            answer = clean_response(answer)
            tokens = response.usage.total_tokens if response.usage else None
            return answer, tokens

        except Exception as e:
            logger.error(f"❌ Erreur Groq API: {e}")
            if "rate_limit" in str(e).lower():
                msg_fr = "⚠️ Limite atteinte. Réessayez dans quelques instants."
                msg_en = "⚠️ Rate limit reached. Please try again."
                return msg_fr if language == "fr" else msg_en, None
            msg_fr = "❌ Erreur lors de la génération. Veuillez réessayer."
            msg_en = "❌ Generation error. Please try again."
            return msg_fr if language == "fr" else msg_en, None


groq_service = GroqService()