import google.generativeai as genai
import requests
import time
from google.api_core import exceptions as google_exceptions

# -------------------------------
# Gemini API Setup
# -------------------------------
# Configure your API key once, e.g. from Streamlit secrets or env
# genai.configure(api_key="YOUR_KEY")

model = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------
# Gemini helpers
# -------------------------------
def get_gemini_response(question, retries=3, delay=2, max_chars=6000):
    """Call Gemini safely with retry, backoff, and size control."""
    if len(question) > max_chars:
        question = question[:max_chars] + "\n...(truncated)..."

    for attempt in range(retries):
        try:
            response = model.generate_content(question)
            return response.text if hasattr(response, "text") else str(response)
        except google_exceptions.ResourceExhausted:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"⚠️ Gemini quota hit. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "⚠️ Gemini API quota exhausted. Please try again later."
        except Exception as e:
            return f"❌ Error: {str(e)}"


def get_gemini_response_with_pdf(user_input, pdf_content, prompt, retries=3):
    """Call Gemini with user input + PDF context + prompt."""
    if not pdf_content:
        return "⚠️ No PDF content provided."

    try:
        response = model.generate_content([user_input, pdf_content[0], prompt])
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"


# -------------------------------
# LeetCode GraphQL API
# -------------------------------
def get_leetcode_data(username):
    """Fetch user profile + submissions + contests from LeetCode."""
    url = "https://leetcode.com/graphql"
    query = """
    query getLeetCodeData($username: String!) {
      userProfile: matchedUser(username: $username) {
        username
        profile {
          userAvatar
          reputation
          ranking
        }
        submitStats {
          acSubmissionNum {
            difficulty
            count
          }
          totalSubmissionNum {
            difficulty
            count
          }
        }
      }
      userContestRanking(username: $username) {
        attendedContestsCount
        rating
        globalRanking
        totalParticipants
        topPercentage
      }
      recentSubmissionList(username: $username) {
        title
        statusDisplay
        lang
      }
      matchedUser(username: $username) {
        languageProblemCount {
          languageName
          problemsSolved
        }
      }
      recentAcSubmissionList(username: $username, limit: 15) {
        id
        title
        titleSlug
        timestamp
      }
    }
    """
    variables = {"username": username}
    response = requests.post(url, json={"query": query, "variables": variables})
    data = response.json()

    if "errors" in data:
        print("Error:", data["errors"])
        return None

    return data.get("data", {})


# -------------------------------
# Load Lottie Animation
# -------------------------------
def load_lottieurl(url: str):
    """Fetch JSON for Lottie animation from URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
