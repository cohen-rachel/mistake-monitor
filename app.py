import streamlit as st
import matplotlib.pyplot as plt
import asyncio
import tempfile
from collections import Counter
import nest_asyncio

# Import your existing pipeline code
from langchain_tutor import AdvancedLanguageTutor

# Enable nested event loops for Streamlit
nest_asyncio.apply()

st.title("Language Learning Tracker")

# Initialize the tutor (use session state to avoid reinitializing)
if 'tutor' not in st.session_state:
    st.session_state.tutor = AdvancedLanguageTutor()

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "aiff"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.aiff') as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
    
    user_id = st.text_input("User ID", value="default_user")
    
    if st.button("Analyze Audio"):
        with st.spinner("Processing audio..."):
            try:
                # Step 1: Transcribe
                st.info("🎤 Transcribing audio...")
                transcription = st.session_state.tutor.transcribe_audio(audio_path)
                
                if not transcription["text"]:
                    st.error("❌ No text transcribed. Please check your audio file.")
                else:
                    st.success(f"✅ Transcribed: {transcription['text']}")
                    
                    # Step 2: Analyze
                    st.info("🤖 Analyzing transcript...")
                    analysis = asyncio.run(st.session_state.tutor.analyze_transcript(transcription["text"]))
                    
                    # Display analysis
                    st.subheader("Analysis Results")
                    st.write(f"**Difficulty Level:** {analysis.difficulty_level}")
                    st.write(f"**Overall Confidence:** {analysis.overall_confidence:.2f}")
                    
                    # Display mistakes
                    st.subheader("Mistakes Found")
                    if analysis.mistakes:
                        for m in analysis.mistakes:
                            st.write(f"- **{m.type}**: \"{m.span}\" → Suggestion: \"{m.suggested_correction}\"")
                            st.write(f"  Explanation: {m.explanation_short}")
                        
                        # Plot mistake distribution
                        mistake_types = [m.type for m in analysis.mistakes]
                        counter = Counter(mistake_types)
                        
                        if counter:
                            fig, ax = plt.subplots()
                            ax.bar(counter.keys(), counter.values())
                            ax.set_ylabel("Frequency")
                            ax.set_title("Most Common Mistakes")
                            plt.xticks(rotation=45, ha='right')
                            st.pyplot(fig)
                        
                        # Generate and show practice exercise
                        st.subheader("Practice Exercise")
                        if st.button("Generate Practice Exercise"):
                            with st.spinner("Generating practice exercise..."):
                                most_common_mistake = max(analysis.mistakes, key=lambda m: m.confidence)
                                exercise = asyncio.run(st.session_state.tutor.generate_practice_exercise(
                                    most_common_mistake.type,
                                    analysis.difficulty_level,
                                    user_id
                                ))
                                
                                st.write(f"**Target:** {exercise.mistake_type}")
                                st.write(f"**Prompt:** {exercise.prompt}")
                                st.write(f"**Expected Answer:** {exercise.expected_answer}")
                                if exercise.hints:
                                    st.write("**Hints:**")
                                    for hint in exercise.hints:
                                        st.write(f"- {hint}")
                    else:
                        st.success("🎉 No mistakes found! Great job!")
                    
                    # Track progress
                    st.subheader("Progress Tracking")
                    progress = asyncio.run(st.session_state.tutor.track_progress(user_id, analysis))
                    st.write(f"**Improvement Score:** {progress.overall_improvement:.2f}")
                    st.write(f"**Next Focus Areas:** {', '.join(progress.next_focus_areas)}")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
