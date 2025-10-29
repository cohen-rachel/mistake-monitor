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

# Initialize session state variables for streaming
if 'streaming_transcript' not in st.session_state:
    st.session_state.streaming_transcript = ""
if 'streaming_chunks' not in st.session_state:
    st.session_state.streaming_chunks = []
if 'streaming_mistakes' not in st.session_state:
    st.session_state.streaming_mistakes = []
if 'all_chunk_mistakes' not in st.session_state:
    st.session_state.all_chunk_mistakes = []  # Store mistakes from all chunks
if 'chunk_counter' not in st.session_state:
    st.session_state.chunk_counter = 0

# Create tabs for different modes
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Stream Live"])

with tab1:
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

with tab2:
    st.header("🎙️ Live Audio Streaming")
    st.markdown("Record audio in real-time and see transcriptions and analysis appear live!")
    
    user_id = st.text_input("User ID", value="default_user", key="streaming_user_id")
    
    # Audio recorder
    audio_bytes = st.audio_input("Record Audio", key="audio_input_stream")
    
    if audio_bytes is not None:
        # Process the audio chunk
        st.session_state.chunk_counter += 1
        current_chunk_num = st.session_state.chunk_counter
        
        with st.spinner(f"Processing chunk {current_chunk_num}..."):
            try:
                # Read bytes from UploadedFile object
                audio_data = audio_bytes.read()
                
                # Transcribe the current chunk
                transcription = st.session_state.tutor.transcribe_audio_streaming(audio_data, format="wav")
                
                if transcription["text"]:
                    # Append to streaming transcript
                    st.session_state.streaming_transcript += " " + transcription["text"]
                    st.session_state.streaming_chunks.append({
                        "text": transcription["text"],
                        "confidence": transcription["confidence_summary"]["avg_token_confidence"],
                        "chunk_num": current_chunk_num
                    })
                    
                    # Show current chunk transcription
                    st.success(f"✅ Chunk {current_chunk_num}: {transcription['text']}")
                    
                    # Analyze THIS chunk independently for immediate feedback
                    if transcription["text"].strip():
                        with st.spinner(f"Analyzing chunk {current_chunk_num}..."):
                            chunk_analysis = asyncio.run(st.session_state.tutor.analyze_transcript(transcription["text"]))
                            
                            # Store mistakes from this chunk
                            if chunk_analysis.mistakes:
                                for mistake in chunk_analysis.mistakes:
                                    st.session_state.all_chunk_mistakes.append({
                                        "chunk": current_chunk_num,
                                        "text": transcription["text"],
                                        "mistake": mistake
                                    })
                                
                                # Show found immediately
                                st.error(f"🔥 Found {len(chunk_analysis.mistakes)} error(s) in chunk {current_chunk_num}!")
                                for mistake in chunk_analysis.mistakes:
                                    with st.expander(f"❌ {mistake.type}: {mistake.span}"):
                                        st.write(f"**From:** Chunk {current_chunk_num}: \"{transcription['text']}\"")
                                        st.write(f"**Should be:** \"{mistake.suggested_correction}\"")
                                        st.write(f"**Explanation:** {mistake.explanation_short}")
                                        st.write(f"**Example:** {mistake.example_correct_sentence}")
                                        st.write(f"**Confidence:** {mistake.confidence:.2f}")
                            else:
                                st.success(f"✓ Chunk {current_chunk_num} has no errors!")
                    
                    # Display running summary
                    st.subheader("📊 Session Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", len(st.session_state.streaming_chunks))
                    with col2:
                        st.metric("Errors Found", len(st.session_state.all_chunk_mistakes))
                    with col3:
                        if st.session_state.streaming_chunks:
                            avg_conf = sum(chunk["confidence"] if chunk["confidence"] else 0 
                                          for chunk in st.session_state.streaming_chunks) / len(st.session_state.streaming_chunks)
                            st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    
                    # Display all mistakes found so far
                    if st.session_state.all_chunk_mistakes:
                        st.subheader("📝 All Errors Found Across Chunks")
                        for idx, error_item in enumerate(st.session_state.all_chunk_mistakes, 1):
                            mistake = error_item["mistake"]
                            st.info(f"**#{idx}** - Chunk {error_item['chunk']}: **{mistake.type}** - \"{mistake.span}\" → \"{mistake.suggested_correction}\"")
                    
                    # Show full transcript
                    st.subheader("📄 Full Transcript")
                    st.text(st.session_state.streaming_transcript)
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Clear button - update to include new state variables
        if st.button("🔄 Clear All Data", key="clear_all"):
            st.session_state.streaming_transcript = ""
            st.session_state.streaming_chunks = []
            st.session_state.streaming_mistakes = []
            st.session_state.all_chunk_mistakes = []
            st.session_state.chunk_counter = 0
            st.success("All streaming data cleared!")
            st.rerun()
        
        # Remove duplicate summary - already shown above in chunk processing
        # Show summary when available
        if len(st.session_state.streaming_chunks) > 0 and False:  # Disabled to avoid duplication
            st.subheader("📊 Session Summary")
            total_chunks = len(st.session_state.streaming_chunks)
            avg_confidence = sum(chunk["confidence"] if chunk["confidence"] else 0 
                                for chunk in st.session_state.streaming_chunks) / total_chunks
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Audio Chunks", total_chunks)
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}" if avg_confidence else "N/A")
            
            # Final analysis button
            if st.button("🎯 Run Final Analysis"):
                with st.spinner("Running comprehensive analysis..."):
                    try:
                        analysis = asyncio.run(st.session_state.tutor.analyze_transcript(st.session_state.streaming_transcript))
                        
                        st.subheader("Final Analysis Results")
                        st.write(f"**Difficulty Level:** {analysis.difficulty_level}")
                        st.write(f"**Overall Confidence:** {analysis.overall_confidence:.2f}")
                        st.write(f"**Total Mistakes:** {len(analysis.mistakes)}")
                        
                        if analysis.mistakes:
                            st.subheader("All Mistakes")
                            for m in analysis.mistakes:
                                with st.expander(f"{m.type}: {m.span}"):
                                    st.write(f"**Correction:** {m.suggested_correction}")
                                    st.write(f"**Explanation:** {m.explanation_short}")
                                    st.write(f"**Example:** {m.example_correct_sentence}")
                                    st.write(f"**Confidence:** {m.confidence:.2f}")
                            
                            # Progress tracking
                            progress = asyncio.run(st.session_state.tutor.track_progress(user_id, analysis))
                            st.subheader("📈 Progress")
                            st.write(f"**Improvement Score:** {progress.overall_improvement:.2f}")
                            st.write(f"**Focus Areas:** {', '.join(progress.next_focus_areas)}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
