import gradio as gr
import sqlite3
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional

# Initialize SQLite database
def init_db():
    # conn = sqlite3.connect('language_learning.db')
    # c = conn.cursor()
    # Create a thread-local storage for the connection
    if not hasattr(init_db, "conn"):
        init_db.conn = sqlite3.connect('language_learning.db', check_same_thread=False)
        c = init_db.conn.cursor()
        # Conversations table
        c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learning_language TEXT,
            known_language TEXT,
            proficiency_level TEXT,
            scenario TEXT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME
        )
        ''')
        
        # Messages table
        c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sender TEXT,
            message TEXT,
            is_correction BOOLEAN DEFAULT 0,
            corrected_text TEXT,
            mistake_type TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        # Mistakes table
        c.execute('''
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_text TEXT,
            corrected_text TEXT,
            mistake_type TEXT,
            explanation TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        # Vocabulary table
        c.execute('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            word TEXT,
            translation TEXT,
            example_sentence TEXT,
            added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')

        init_db.conn.commit()
        # conn.commit()
    # return conn
    return init_db.conn

# Initialize database connection
conn = init_db()

class LanguageLearningChatbot:
    def __init__(self):
        self.current_conversation: Optional[int] = None
        self.conversation_chain: Optional[LLMChain] = None
        self.messages: List[Dict[str, str]] = []  # Changed to use dict format
        
        # Define available languages including Indian languages
        self.languages = [
            "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Urdu", "Punjabi",
            "Spanish", "French", "German", "Italian", "Japanese", "Chinese", "Russian", "Portuguese", "English"
        ]
        
        # Indian language scripts mapping (for display purposes)
        self.scripts = {
            "Hindi": "Devanagari",
            "Bengali": "Bengali",
            "Tamil": "Tamil",
            "Telugu": "Telugu",
            "Marathi": "Devanagari",
            "Gujarati": "Gujarati",
            "Urdu": "Perso-Arabic",
            "Punjabi": "Gurmukhi"
        }
        
        self.proficiency_levels = [
            "A1 Beginner", "A2 Elementary", "B1 Intermediate", 
            "B2 Upper Intermediate", "C1 Advanced", "C2 Proficient"
        ]
        
        # Scenarios with Indian context
        self.scenarios = [
            "At a restaurant", "Asking for directions", "Market shopping",
            "Train station", "Doctor's visit", "Family gathering",
            "Festival celebration", "Hotel check-in", "Job interview", "Custom"
        ]
        
        # Create Gradio interface
        self.create_interface()
    
    def create_interface(self):
        with gr.Blocks(title="Language Learning Chatbot", theme=gr.themes.Soft()) as self.demo:
            gr.Markdown("# 🌍 Language Learning Chatbot (with Indian Languages)")
            
            with gr.Tab("New Conversation"):
                with gr.Row():
                    with gr.Column():
                        self.learning_lang = gr.Dropdown(
                            label="Language you want to learn",
                            choices=self.languages,
                            value="Hindi"
                        )
                        self.proficiency = gr.Dropdown(
                            label="Your current proficiency level",
                            choices=self.proficiency_levels,
                            value="A1 Beginner"
                        )
                        self.script_info = gr.Markdown("")
                    with gr.Column():
                        self.known_lang = gr.Dropdown(
                            label="Language you know well",
                            choices=self.languages,
                            value="English"
                        )
                        self.scenario = gr.Dropdown(
                            label="Choose a practice scenario",
                            choices=self.scenarios,
                            value="Market shopping"
                        )
                        self.custom_scenario = gr.Textbox(
                            label="Custom scenario (if selected)",
                            visible=False
                        )
                
                self.start_btn = gr.Button("Start Conversation", variant="primary")
                self.status = gr.Markdown("Select options and start a new conversation.")
                
                # Update script info when language changes
                self.learning_lang.change(
                    self.update_script_info,
                    inputs=[self.learning_lang],
                    outputs=[self.script_info]
                )
                
                # Show/hide custom scenario
                self.scenario.change(
                    lambda x: gr.update(visible=x == "Custom"),
                    inputs=[self.scenario],
                    outputs=[self.custom_scenario]
                )
            
            with gr.Tab("Chat"):
                # Updated to use the new messages format
                self.chat_display = gr.Chatbot(label="Conversation", type="messages")
                self.user_input = gr.Textbox(label="Type your message...", placeholder="Type in the language you're learning")
                self.send_btn = gr.Button("Send", variant="primary")
                self.end_btn = gr.Button("End Conversation")
                self.conversation_info = gr.Markdown("No active conversation.")
            
            with gr.Tab("Analysis"):
                with gr.Row():
                    with gr.Column():
                        self.mistakes_df = gr.Dataframe(
                            label="Mistakes",
                            headers=["What you said", "Correction", "Mistake Type", "Explanation"],
                            interactive=False
                        )
                    with gr.Column():
                        self.mistakes_plot = gr.Plot(label="Mistake Distribution")
                
                with gr.Row():
                    with gr.Column():
                        self.vocab_df = gr.Dataframe(
                            label="New Vocabulary",
                            headers=["Word", "Translation", "Example"],
                            interactive=False
                        )
                    with gr.Column():
                        self.recommendations = gr.Markdown("## Areas to Focus On\nStart a conversation to get recommendations.")
            
            with gr.Tab("History"):
                self.conversation_history = gr.DataFrame(
                    label="Past Conversations",
                    headers=["ID", "Learning", "Known", "Level", "Scenario", "Date"],
                    interactive=False
                )
                self.load_history_btn = gr.Button("Refresh History")
                self.delete_conversation_id = gr.Dropdown(
                    label="Select conversation to delete",
                    choices=[]
                )
                self.delete_btn = gr.Button("Delete Conversation", variant="stop")
            
            # Event handlers
            self.start_btn.click(
                self.start_conversation,
                inputs=[self.learning_lang, self.known_lang, self.proficiency, self.scenario, self.custom_scenario],
                outputs=[self.status, self.conversation_info, self.chat_display]
            )
            
            self.send_btn.click(
                self.send_message,
                inputs=[self.user_input],
                outputs=[self.user_input, self.chat_display, self.mistakes_df, self.vocab_df, self.mistakes_plot, self.recommendations]
            )
            
            self.user_input.submit(
                self.send_message,
                inputs=[self.user_input],
                outputs=[self.user_input, self.chat_display, self.mistakes_df, self.vocab_df, self.mistakes_plot, self.recommendations]
            )
            
            self.end_btn.click(
                self.end_conversation,
                outputs=[self.conversation_info, self.chat_display]
            )
            
            self.load_history_btn.click(
                self.load_history,
                outputs=[self.conversation_history, self.delete_conversation_id]
            )
            
            self.delete_btn.click(
                self.delete_conversation_handler,
                inputs=[self.delete_conversation_id],
                outputs=[self.conversation_history, self.delete_conversation_id]
            )
            
            # Initialize
            self.load_history()
            self.update_script_info(self.learning_lang.value)
    


    def update_script_info(self, language: str) -> Dict:
        """Update the script information display based on selected language"""
        if language in self.scripts:
            return {"value": f"**Script:** {self.scripts[language]}", "__type__": "update"}
        return {"value": "", "__type__": "update"}
    
    def init_conversation(self, learning_lang: str, known_lang: str, proficiency: str, scenario: str) -> LLMChain:
        """Initialize the LangChain conversation chain with Azure o3-mini model"""
       

        DEEPSEEK_API_KEY = " "  # Replace with your Deepseek Openrouter API key
        DEEPSEEK_API_BASE = "https://openrouter.ai/api/v1"
        
        llm = ChatOpenAI(
            model_name="deepseek/deepseek-chat-v3-0324:free",
            temperature=1,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            streaming=False
        )
        
        # Enhanced prompt with Indian language considerations
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are a friendly {learning_lang} language teacher. The student knows {known_lang} and their 
            proficiency level in {learning_lang} is {proficiency}. You are currently practicing a 
            scenario about: {scenario}.
            
            Rules:
            1. Conduct the conversation primarily in {learning_lang}.
            2. For Indian languages, provide transliterations in Latin script for beginners.
            3. For beginner levels (A1-A2), use simple vocabulary and short sentences.
            4. For intermediate levels (B1-B2), use more complex structures but still keep it understandable.
            5. For advanced levels (C1-C2), speak naturally with complex structures.
            6. Correct mistakes gently by first repeating the corrected version, then briefly explaining.
            7. Keep track of mistakes in a structured way.
            8. Occasionally introduce relevant vocabulary with translations.
            9. For Indian contexts, use culturally appropriate examples.
            10. Be encouraging and positive.
            
            Additional Guidelines for Indian Languages:
            - For Hindi: Use Devanagari script but provide Roman transliteration when needed
            - For South Indian languages: Pay attention to proper noun endings
            - For Bengali: Note the different verb conjugations
            - For Urdu: Include both Perso-Arabic script and Roman transliteration
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        memory = ConversationBufferMemory(return_messages=True)
        return LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=True
        )
    
    

    def save_conversation(self, learning_lang: str, known_lang: str, proficiency: str, scenario: str) -> int:
        """Save new conversation to database"""
        conn = init_db()  # Get connection from thread-safe storage
        c = conn.cursor()
        c.execute('''
        INSERT INTO conversations (learning_language, known_language, proficiency_level, scenario)
        VALUES (?, ?, ?, ?)
        ''', (learning_lang, known_lang, proficiency, scenario))
        conn.commit()
        return c.lastrowid
    
    def start_conversation(self, learning_lang: str, known_lang: str, proficiency: str, scenario: str, custom_scenario: str) -> Tuple[Dict, Dict, List]:
        """Start a new conversation"""
        if scenario == "Custom" and custom_scenario:
            scenario = custom_scenario
        
        # Initialize conversation
        self.current_conversation = self.save_conversation(learning_lang, known_lang, proficiency, scenario)
        self.conversation_chain = self.init_conversation(learning_lang, known_lang, proficiency, scenario)
        self.messages = []
        
        # Add welcome message with script info if Indian language
        welcome_msg = f"Let's practice {learning_lang}!"
        if learning_lang in self.scripts:
            welcome_msg += f" (Script: {self.scripts[learning_lang]})"
        welcome_msg += f"\nWe'll simulate: {scenario}. I'll help correct your mistakes."
        
        # Updated to use the new message format
        self.messages.append({"role": "assistant", "content": welcome_msg})
        self.save_message(self.current_conversation, "assistant", welcome_msg)
        
        # Update UI
        status = f"Started new conversation: Learning {learning_lang} (know {known_lang}, level {proficiency}), scenario: {scenario}"
        info = f"""### Current Conversation
- **Learning**: {learning_lang} {f"({self.scripts.get(learning_lang, '')})" if learning_lang in self.scripts else ""}
- **From**: {known_lang}
- **Level**: {proficiency}
- **Scenario**: {scenario}"""
        
        return status, info, self.messages
    
    def send_message(self, user_input: str) -> Tuple[str, List, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[plt.Figure], str]:
        """Process and respond to user message"""
        if not self.current_conversation:
            return "", self.messages, None, None, None, "No active conversation. Please start one first."
        
        # Add user message (updated format)
        self.messages.append({"role": "user", "content": user_input})
        self.save_message(self.current_conversation, "user", user_input)
        
        # Get AI response
        response = self.conversation_chain.run(input=user_input)
        
        # Process response for mistakes and vocabulary
        if "Correction:" in response:
            parts = response.split("Correction:")
            main_response = parts[0]
            correction_part = parts[1]
            
            if "Explanation:" in correction_part:
                correction, explanation = correction_part.split("Explanation:")
                original_text = user_input
                corrected_text = correction.strip()
                explanation = explanation.strip()
                
                # Determine mistake type
                mistake_type = "grammar"
                if "vocabulary" in explanation.lower():
                    mistake_type = "vocabulary"
                elif "pronunciation" in explanation.lower():
                    mistake_type = "pronunciation"
                elif "word order" in explanation.lower():
                    mistake_type = "word order"
                elif "script" in explanation.lower():
                    mistake_type = "script"
                
                # Save mistake
                self.save_mistake(
                    self.current_conversation,
                    original_text,
                    corrected_text,
                    mistake_type,
                    explanation
                )
                
                # Save correction message
                self.save_message(
                    self.current_conversation,
                    "assistant",
                    response,
                    True,
                    corrected_text,
                    mistake_type
                )
        else:
            self.save_message(
                self.current_conversation,
                "assistant",
                response
            )
            
            # Check for vocabulary introduction
            if "Vocabulary:" in response:
                vocab_part = response.split("Vocabulary:")[1].split("\n")[0]
                if "-" in vocab_part:
                    word, translation = vocab_part.split("-", 1)
                    example = response.split("Example:")[1].split("\n")[0] if "Example:" in response else ""
                    self.save_vocabulary(
                        self.current_conversation,
                        word.strip(),
                        translation.strip(),
                        example.strip()
                    )
        
        # Add AI response to chat (updated format)
        self.messages.append({"role": "assistant", "content": response})
        
        # Get updated analysis data
        mistakes_df, vocab_df, plot, recommendations = self.get_analysis_data()
        
        return "", self.messages, mistakes_df, vocab_df, plot, recommendations
    
    def save_message(self, conversation_id: int, sender: str, message: str, 
                    is_correction: bool = False, corrected_text: Optional[str] = None, 
                    mistake_type: Optional[str] = None) -> None:
        """Save message to database"""
        c = conn.cursor()
        c.execute('''
        INSERT INTO messages (conversation_id, sender, message, is_correction, corrected_text, mistake_type)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, sender, message, is_correction, corrected_text, mistake_type))
        conn.commit()
    
    def save_mistake(self, conversation_id: int, original_text: str, corrected_text: str, 
                    mistake_type: str, explanation: str) -> None:
        """Save mistake to database"""
        c = conn.cursor()
        c.execute('''
        INSERT INTO mistakes (conversation_id, original_text, corrected_text, mistake_type, explanation)
        VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, original_text, corrected_text, mistake_type, explanation))
        conn.commit()
    
    def save_vocabulary(self, conversation_id: int, word: str, translation: str, 
                       example_sentence: str) -> None:
        """Save vocabulary to database"""
        c = conn.cursor()
        c.execute('''
        INSERT INTO vocabulary (conversation_id, word, translation, example_sentence)
        VALUES (?, ?, ?, ?)
        ''', (conversation_id, word, translation, example_sentence))
        conn.commit()
    
    def get_analysis_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[plt.Figure], str]:
        """Get analysis data for current conversation"""
        if not self.current_conversation:
            return None, None, None, "No active conversation"
        
        # Get mistakes
        mistakes = self.get_mistakes(self.current_conversation)
        if mistakes:
            mistakes_df = pd.DataFrame(mistakes, columns=["What you said", "Correction", "Mistake Type", "Explanation"])
            
            # Create plot
            mistake_counts = mistakes_df['Mistake Type'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(mistake_counts, labels=mistake_counts.index, autopct='%1.1f%%')
            ax.set_title("Mistake Type Distribution")
            
            # Create recommendations
            recommendations = "## Areas to Focus On\n"
            if "grammar" in mistake_counts:
                recommendations += "- 📝 **Grammar**: Practice verb conjugations and sentence structure.\n"
            if "vocabulary" in mistake_counts:
                recommendations += "- 📖 **Vocabulary**: Review flashcards and try to use new words in sentences.\n"
            if "pronunciation" in mistake_counts:
                recommendations += "- 🎤 **Pronunciation**: Listen to native speakers and repeat after them.\n"
            if "word order" in mistake_counts:
                recommendations += "- 🔠 **Word Order**: Practice constructing sentences with different structures.\n"
            if "script" in mistake_counts:
                recommendations += "- ✍️ **Script**: Practice writing characters/letters of the alphabet.\n"
        else:
            mistakes_df = pd.DataFrame(columns=["What you said", "Correction", "Mistake Type", "Explanation"])
            fig = plt.figure()
            plt.text(0.5, 0.5, "No mistakes yet!", ha='center', va='center')
            recommendations = "## Areas to Focus On\nNo mistakes recorded yet. Keep practicing!"
        
        # Get vocabulary
        vocab = self.get_vocabulary(self.current_conversation)
        if vocab:
            vocab_df = pd.DataFrame(vocab, columns=["Word", "Translation", "Example"])
        else:
            vocab_df = pd.DataFrame(columns=["Word", "Translation", "Example"])
        
        return mistakes_df, vocab_df, fig, recommendations
    
    def get_conversations(self) -> List[Tuple]:
        """Get all conversations from database"""
        c = conn.cursor()
        return c.execute('''
        SELECT id, learning_language, known_language, proficiency_level, scenario, start_time 
        FROM conversations 
        ORDER BY start_time DESC
        ''').fetchall()
    
    def get_mistakes(self, conversation_id: int) -> List[Tuple]:
        """Get mistakes for a conversation"""
        c = conn.cursor()
        return c.execute('''
        SELECT original_text, corrected_text, mistake_type, explanation 
        FROM mistakes 
        WHERE conversation_id = ? 
        ORDER BY timestamp
        ''', (conversation_id,)).fetchall()
    
    def get_vocabulary(self, conversation_id: int) -> List[Tuple]:
        """Get vocabulary for a conversation"""
        c = conn.cursor()
        return c.execute('''
        SELECT word, translation, example_sentence 
        FROM vocabulary 
        WHERE conversation_id = ? 
        ORDER BY added_date
        ''', (conversation_id,)).fetchall()
    
    def end_conversation(self) -> Tuple[Dict, List]:
        """End current conversation"""
        if self.current_conversation:
            # Update end time in database
            c = conn.cursor()
            c.execute('''
            UPDATE conversations 
            SET end_time = CURRENT_TIMESTAMP 
            WHERE id = ?
            ''', (self.current_conversation,))
            conn.commit()
            
            # Reset state
            self.current_conversation = None
            self.conversation_chain = None
            self.messages = []
            
            return "Conversation ended. Start a new one to continue learning.", []
        return "No active conversation to end.", []
    
    def load_history(self) -> Tuple[List[Tuple], Dict]:
        """Load conversation history"""
        conversations = self.get_conversations()
        if conversations:
            # Format for display
            display_data = [
                (conv[0], conv[1], conv[2], conv[3], conv[4], conv[5].split()[0])
                for conv in conversations
            ]
            
            # Update delete dropdown
            delete_options = [str(conv[0]) for conv in conversations]
            
            return display_data, {"choices": delete_options, "__type__": "update"}
        return [], {"choices": [], "__type__": "update"}
    
    def delete_conversation_handler(self, conversation_id: str) -> Tuple[List[Tuple], Dict]:
        """Handle conversation deletion"""
        if conversation_id:
            self.delete_conversation(int(conversation_id))
            return self.load_history()
        return self.load_history()
    
    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation from database"""
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM mistakes WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM vocabulary WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()

# Run the application
if __name__ == "__main__":
    # Set your GitHub Token as environment variable
    # export GITHUB_TOKEN='your-token-here'
    
    chatbot = LanguageLearningChatbot()
    chatbot.demo.launch()
