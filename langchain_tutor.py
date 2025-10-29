#!/usr/bin/env python3
"""
Advanced Language Tutor with LangChain
Features:
- Conversational practice mode
- Progress tracking with vector storage
- Multi-step analysis pipeline
- Personalized learning paths
- Memory of user mistakes
"""
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.callbacks import StreamingStdOutCallbackHandler

from pydantic import BaseModel, Field

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2:latest"
WHISPER_MODEL = "base"
VECTOR_DB_PATH = "./vector_db"

class Mistake(BaseModel):
    """Structured mistake representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(description="Type of mistake: verb-tense, preposition, article, etc.")
    label: str = Field(description="Human-readable label")
    span: str = Field(description="The incorrect text span")
    start_char: int = Field(description="Start position in transcript")
    end_char: int = Field(description="End position in transcript")
    suggested_correction: str = Field(description="Corrected version")
    explanation_short: str = Field(description="Brief explanation")
    confidence: float = Field(description="Confidence score 0-1")
    example_correct_sentence: str = Field(description="Example of correct usage")
    stt_uncertain: bool = Field(default=False, description="Whether STT was uncertain")
    uncertain: bool = Field(default=False, description="Whether analysis was uncertain")

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    session_id: str
    language: str = "en"
    transcript: str
    mistakes: List[Mistake]
    overall_confidence: float = Field(description="Overall analysis confidence")
    difficulty_level: str = Field(description="Estimated difficulty: beginner, intermediate, advanced")

class PracticeExercise(BaseModel):
    """Practice exercise structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(description="Exercise prompt")
    expected_answer: str = Field(description="Expected correct answer")
    mistake_type: str = Field(description="Targeted mistake type")
    difficulty: str = Field(description="Exercise difficulty")
    hints: List[str] = Field(default_factory=list, description="Helpful hints")

class UserProgress(BaseModel):
    """User progress tracking"""
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    mistakes_made: List[str] = Field(description="Types of mistakes made")
    mistakes_corrected: List[str] = Field(description="Types of mistakes corrected")
    overall_improvement: float = Field(description="Improvement score 0-1")
    next_focus_areas: List[str] = Field(description="Areas to focus on next")

class AdvancedLanguageTutor:
    def __init__(self):
        print("🚀 Initializing Advanced Language Tutor with LangChain...")
        
        self.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        self.vectorstore = self._init_vectorstore()
        
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        self._init_chains()
        
        print("✅ Advanced Language Tutor initialized!")

    def _init_vectorstore(self):
        """Initialize vector store for progress tracking"""
        try:
            if os.path.exists(VECTOR_DB_PATH):
                return Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
            else:
                return Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"⚠️  Vector store initialization failed: {e}")
            return None

    def _init_chains(self):
        """Initialize LangChain chains for different operations"""
        
        analysis_examples = [
            {
                "transcript": "I goed to the store yesterday.",
                "mistakes": "verb-tense: 'goed' -> 'went' (Irregular past tense of 'go' is 'went')"
            },
            {
                "transcript": "She have many books.",
                "mistakes": "verb-agreement: 'have' -> 'has' (Third person singular requires 'has' not 'have')"
            }
        ]
        
        # Analysis prompt with few-shot examples
        analysis_prompt = FewShotPromptTemplate(
            examples=analysis_examples,
            example_prompt=PromptTemplate(
                input_variables=["transcript", "mistakes"],
                template="Transcript: {transcript}\nMistakes: {mistakes}"
            ),
            prefix="""You are an expert English language tutor. Analyze transcripts for common learner mistakes.
            Focus on: verb-tense, verb-agreement, prepositions, articles, word-order, pronouns, pluralization.
            
            Examples:""",
            suffix="""
            Now analyze this transcript: {transcript}
            
            Return a JSON object with this structure:
            {{
                "session_id": "{{session_id}}",
                "language": "en",
                "transcript": "{{transcript}}",
                "mistakes": [
                    {{
                        "type": "verb-tense",
                        "label": "Verb tense",
                        "span": "incorrect_word",
                        "start_char": 0,
                        "end_char": 10,
                        "suggested_correction": "correct_word",
                        "explanation_short": "Brief explanation",
                        "confidence": 0.9,
                        "example_correct_sentence": "Example sentence",
                        "stt_uncertain": false,
                        "uncertain": false
                    }}
                ],
                "overall_confidence": 0.8,
                "difficulty_level": "intermediate"
            }}
            """,
            input_variables=["transcript", "session_id"]
        )
        
        # practice generation prompt
        practice_prompt = PromptTemplate(
            input_variables=["mistake_type", "difficulty", "user_history"],
            template="""
            Generate a practice exercise for mistake type: {mistake_type}
            Difficulty level: {difficulty}
            User's common mistakes: {user_history}
            
            Create an exercise that:
            1. Targets the specific mistake type
            2. Matches the difficulty level
            3. Avoids the user's common error patterns
            
            Return JSON:
            {{
                "prompt": "Exercise instruction",
                "expected_answer": "Correct answer",
                "mistake_type": "{mistake_type}",
                "difficulty": "{difficulty}",
                "hints": ["hint1", "hint2"]
            }}
            """
        )
        
        # conversational practice prompt
        conversation_prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""
            You are a patient English tutor having a conversation with a language learner.
            Previous conversation: {history}
            
            Student says: {input}
            
            Respond as a helpful tutor:
            1. Acknowledge what they said
            2. Gently correct any mistakes
            3. Ask a follow-up question to continue practice
            4. Keep responses encouraging and supportive
            
            Focus on natural conversation while providing learning opportunities.
            """
        )
        
        # init chains
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analysis_prompt,
            output_key="analysis"
        )
        
        self.practice_chain = LLMChain(
            llm=self.llm,
            prompt=practice_prompt,
            output_key="practice"
        )
        
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.conversation_memory,
            prompt=conversation_prompt
        )

    def transcribe_audio(self, audio_path: str) -> dict:
        """Transcribe audio using faster-whisper"""
        try:
            from faster_whisper import WhisperModel
            print(f"🎤 Transcribing: {audio_path}")
            
            model = WhisperModel(WHISPER_MODEL, device="cpu")
            segments, info = model.transcribe(audio_path, language="en", beam_size=1)
            
            words = []
            full_text_parts = []
            
            for seg in segments:
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        words.append({
                            "word": w.word,
                            "start": float(w.start) if w.start is not None else None,
                            "end": float(w.end) if w.end is not None else None,
                            "confidence": float(w.probability) if getattr(w, 'probability', None) is not None else None,
                        })
                        full_text_parts.append(w.word)
                else:
                    full_text_parts.append(seg.text)
            
            text = " ".join([t.strip() for t in full_text_parts]).strip()
            
            avg_conf = None
            confs = [w["confidence"] for w in words if w.get("confidence") is not None]
            if confs:
                avg_conf = sum(confs) / len(confs)
            
            result = {
                "text": text,
                "tokens": words,
                "confidence_summary": {"avg_token_confidence": avg_conf},
            }
            
            print(f"✅ Transcription complete: {len(text)} characters")
            print(f"📝 Text: {text}")
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return {"text": "", "tokens": [], "confidence_summary": {"avg_token_confidence": None}}

    async def analyze_transcript(self, transcript: str, session_id: str = None) -> AnalysisResult:
        """Analyze transcript using LangChain pipeline"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        print("🤖 Analyzing with LangChain pipeline...")
        
        try:
            
            result = await self.analysis_chain.ainvoke(transcript)
            
            raw_response = result['analysis']
            analysis_data = self._parse_json_response(raw_response)
            
            # Convert to Pydantic model
            mistakes = [Mistake(**mistake) for mistake in analysis_data.get("mistakes", [])]
            
            analysis_result = AnalysisResult(
                session_id=session_id,
                language=analysis_data.get("language", "en"),
                transcript=transcript,
                mistakes=mistakes,
                overall_confidence=analysis_data.get("overall_confidence", 0.8),
                difficulty_level=analysis_data.get("difficulty_level", "intermediate")
            )
            
            print(f"✅ Analysis complete: {len(mistakes)} mistakes found")
            return analysis_result
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return AnalysisResult(
                session_id=session_id,
                transcript=transcript,
                mistakes=[],
                overall_confidence=0.0,
                difficulty_level="unknown"
            )

    async def generate_practice_exercise(self, mistake_type: str, difficulty: str = "intermediate", user_id: str = "default") -> PracticeExercise:
        """Generate personalized practice exercise"""
        print(f"📚 Generating practice for {mistake_type} at {difficulty} level...")
        
        user_history = await self.get_user_mistake_history(user_id)
        
        try:
            result = await self.practice_chain.arun(
                mistake_type=mistake_type,
                difficulty=difficulty,
                user_history=user_history
            )
            
            exercise_data = json.loads(result)
            
            exercise = PracticeExercise(
                prompt=exercise_data["prompt"],
                expected_answer=exercise_data["expected_answer"],
                mistake_type=mistake_type,
                difficulty=difficulty,
                hints=exercise_data.get("hints", [])
            )
            
            print("✅ Practice exercise generated")
            return exercise
            
        except Exception as e:
            print(f"❌ Practice generation failed: {e}")
            return PracticeExercise(
                prompt=f"Practice {mistake_type}",
                expected_answer="Sample answer",
                mistake_type=mistake_type,
                difficulty=difficulty
            )

    async def conversational_practice(self, user_input: str) -> str:
        """Engage in conversational practice"""
        print("💬 Conversational practice...")
        
        try:
            response = await self.conversation_chain.arun(input=user_input)
            print("✅ Conversation response generated")
            return response
        except Exception as e:
            print(f"❌ Conversation failed: {e}")
            return "I'm sorry, I couldn't process that. Could you try again?"

    async def track_progress(self, user_id: str, analysis: AnalysisResult) -> UserProgress:
        """Track user progress and store in vector database"""
        print("📊 Tracking progress...")
        
        # Extract mistake types
        mistake_types = [mistake.type for mistake in analysis.mistakes]
        
        progress_doc = Document(
            page_content=f"Session {analysis.session_id}: {analysis.transcript}",
            metadata={
                "user_id": user_id,
                "session_id": analysis.session_id,
                "timestamp": datetime.now().isoformat(),
                "mistake_types": ",".join(mistake_types),  # Convert list to comma-separated string
                "difficulty_level": analysis.difficulty_level,
                "overall_confidence": analysis.overall_confidence
            }
        )
        
        # store in vector database
        if self.vectorstore:
            try:
                self.vectorstore.add_documents([progress_doc])
                self.vectorstore.persist()
                print("✅ Progress stored in vector database")
            except Exception as e:
                print(f"⚠️  Failed to store progress: {e}")
        
        # calculate improvement (simplified)
        improvement = max(0, 1 - (len(mistake_types) * 0.1))
        
        progress = UserProgress(
            user_id=user_id,
            session_id=analysis.session_id,
            mistakes_made=mistake_types,
            mistakes_corrected=[],  # would need user feedback to determine
            overall_improvement=improvement,
            next_focus_areas=self._get_next_focus_areas(mistake_types)
        )
        
        return progress

    async def get_user_mistake_history(self, user_id: str) -> str:
        """Get user's mistake history from vector database"""
        if not self.vectorstore:
            return "No history available"
        
        try:
            # search for user's previous sessions
            docs = self.vectorstore.similarity_search(
                f"user {user_id} mistakes",
                filter={"user_id": user_id},
                k=5
            )
            
            if not docs:
                return "No previous mistakes found"
            
            # extract mistake types from metadata
            all_mistakes = []
            for doc in docs:
                mistake_types_str = doc.metadata.get("mistake_types", "")
                # convert comma-separated string back to list
                if mistake_types_str:
                    all_mistakes.extend(mistake_types_str.split(","))
            
            # count frequency
            from collections import Counter
            mistake_counts = Counter(all_mistakes)
            
            return f"Common mistakes: {dict(mistake_counts.most_common(3))}"
            
        except Exception as e:
            print(f"⚠️  Failed to get user history: {e}")
            return "History unavailable"

    def _parse_json_response(self, raw_response: str) -> dict:
        """Parse JSON response with error handling for malformed responses"""
        try:
            # first try to parse as-is
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            
            # try to fix common JSON issues
            try:
                # remove leading/trailing whitespace and newlines
                cleaned = raw_response.strip()
                
                # if it doesn't end with }, try to add it
                if not cleaned.endswith('}'):
                    cleaned += '}'
                
                # try parsing again
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # if still failing, try to extract just the mistakes array
                try:
                    # look for the mistakes array in the response
                    import re
                    mistakes_match = re.search(r'"mistakes":\s*\[(.*?)\]', cleaned, re.DOTALL)
                    if mistakes_match:
                        mistakes_str = mistakes_match.group(1)
                        # try to parse individual mistake objects
                        mistakes = []
                        # simple regex to find mistake objects
                        mistake_pattern = r'\{[^{}]*"type"[^{}]*\}'
                        for match in re.finditer(mistake_pattern, mistakes_str):
                            try:
                                mistake_obj = json.loads(match.group(0))
                                mistakes.append(mistake_obj)
                            except:
                                continue
                        
                        return {
                            "session_id": "fallback",
                            "language": "en",
                            "transcript": "fallback",
                            "mistakes": mistakes,
                            "overall_confidence": 0.5,
                            "difficulty_level": "intermediate"
                        }
                except:
                    pass
                
                # final fallback - return empty result
                print("❌ Could not parse JSON response, returning empty result")
                return {
                    "session_id": "fallback",
                    "language": "en", 
                    "transcript": "fallback",
                    "mistakes": [],
                    "overall_confidence": 0.0,
                    "difficulty_level": "unknown"
                }

    def _get_next_focus_areas(self, current_mistakes: List[str]) -> List[str]:
        """Determine next focus areas based on current mistakes"""
        from collections import Counter
        mistake_counts = Counter(current_mistakes)
        
        # return top 2 most common mistake types
        return [mistake for mistake, count in mistake_counts.most_common(2)]

    def print_analysis(self, analysis: AnalysisResult):
        """Print analysis results in a nice format"""
        print(f"\n📋 Analysis Results (Session: {analysis.session_id})")
        print(f"🎯 Difficulty Level: {analysis.difficulty_level}")
        print(f"📊 Overall Confidence: {analysis.overall_confidence:.2f}")
        print("=" * 60)
        
        if not analysis.mistakes:
            print("🎉 No mistakes found! Great job!")
            return
        
        print(f"Found {len(analysis.mistakes)} correction(s):")
        
        for i, mistake in enumerate(analysis.mistakes, 1):
            print(f"\n{i}. {mistake.label} ({mistake.type})")
            print(f"   Text: \"{mistake.span}\"")
            print(f"   💡 Suggestion: \"{mistake.suggested_correction}\"")
            print(f"   📚 Explanation: {mistake.explanation_short}")
            print(f"   ✅ Example: {mistake.example_correct_sentence}")
            print(f"   🎯 Confidence: {mistake.confidence:.2f}")
            
            if mistake.stt_uncertain:
                print(f"   ⚠️  Note: STT was uncertain about this word")
            if mistake.uncertain:
                print(f"   ❓ Note: Analysis was uncertain about this error")

    def print_practice_exercise(self, exercise: PracticeExercise):
        """Print practice exercise in a nice format"""
        print(f"\n📚 Practice Exercise")
        print(f"🎯 Target: {exercise.mistake_type}")
        print(f"📊 Difficulty: {exercise.difficulty}")
        print("=" * 40)
        print(f"📝 {exercise.prompt}")
        print(f"💡 Expected Answer: {exercise.expected_answer}")
        
        if exercise.hints:
            print("💭 Hints:")
            for hint in exercise.hints:
                print(f"   • {hint}")

async def main():
    """Main function for testing the advanced tutor"""
    if len(sys.argv) < 2:
        print("Usage: python langchain_tutor.py <audio_file> [user_id]")
        print("       python langchain_tutor.py --conversation [user_id]")
        sys.exit(1)
    
    user_id = sys.argv[2] if len(sys.argv) > 2 else "default_user"
    
    if sys.argv[1] == "--conversation":
        # conversational practice mode
        print("💬 Starting conversational practice mode...")
        print("Type 'quit' to exit")
        
        tutor = AdvancedLanguageTutor()
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Keep practicing!")
                break
            
            response = await tutor.conversational_practice(user_input)
            print(f"Tutor: {response}")
    
    else:
        # Audio analysis mode
        audio_path = sys.argv[1]
        
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            sys.exit(1)
        
        print("🎯 Advanced Language Tutor with LangChain")
        print("=" * 50)
        
        # Initialize tutor
        tutor = AdvancedLanguageTutor()
        
        # Step 1: Transcribe
        transcription = tutor.transcribe_audio(audio_path)
        
        if not transcription["text"]:
            print("❌ No text transcribed. Check audio file.")
            sys.exit(1)
        
        # Step 2: Analyze
        analysis = await tutor.analyze_transcript(transcription["text"])
        
        # Step 3: Print results
        tutor.print_analysis(analysis)
        
        # Step 4: Generate practice exercise for most common mistake
        if analysis.mistakes:
            most_common_mistake = max(analysis.mistakes, key=lambda m: m.confidence)
            exercise = await tutor.generate_practice_exercise(
                most_common_mistake.type,
                analysis.difficulty_level,
                user_id
            )
            tutor.print_practice_exercise(exercise)
        
        # Step 5: Track progress
        progress = await tutor.track_progress(user_id, analysis)
        print(f"\n📊 Progress Summary:")
        print(f"   Improvement Score: {progress.overall_improvement:.2f}")
        print(f"   Next Focus Areas: {', '.join(progress.next_focus_areas)}")
        
        print("\n" + "=" * 50)
        print("✅ Advanced analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
