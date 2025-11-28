with col1:
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a plant leaf image**")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG • Max 200MB",
        label_visibility="collapsed"
    )

    # Store file data in session state immediately
    if uploaded_file is not None:
        st.session_state.uploaded_file_data = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name

    # Nice empty state
    if uploaded_file is None and st.session_state.uploaded_file_data is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
        </div>
        """, unsafe_allow_html=True)

    # Process uploaded file from session state data
    if st.session_state.uploaded_file_data is not None:
        try:
            # Create a fresh image object from stored data for display
            image_data = BytesIO(st.session_state.uploaded_file_data)
            display_image = Image.open(image_data)

            st.success("✅ **File uploaded successfully!**")
            st.write(f"**Filename:** {st.session_state.uploaded_file_name}")

            # Preview
            st.image(display_image, caption="📷 Your Plant Leaf", width=400)

            # File info
            file_size_mb = len(st.session_state.uploaded_file_data) / (1024 * 1024)
            st.write(
                f"**Image Details:** {display_image.size[0]} × {display_image.size[1]} pixels • {file_size_mb:.1f} MB"
            )

            # Analyze button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                # 🆕 TEMPORARILY CLEAR HISTORY FOR ACCURATE TESTING
                current_prediction_history = st.session_state.prediction_history.copy()
                st.session_state.prediction_history = []  # Clear history for testing
                
                with st.spinner("🔬 Analyzing your plant..."):
                    # 🆕 CRITICAL FIX: Create a FRESH image object for prediction
                    prediction_image_data = BytesIO(st.session_state.uploaded_file_data)
                    prediction_image = Image.open(prediction_image_data)
                    
                    disease, confidence, error = predict_image(
                        prediction_image, model, class_names, img_size
                    )

                # 🆕 RESTORE HISTORY AFTER PREDICTION
                st.session_state.prediction_history = current_prediction_history
                if disease:
                    st.session_state.prediction_history.append(disease)
                    if len(st.session_state.prediction_history) > 5:
                        st.session_state.prediction_history.pop(0)

                if error:
                    st.error(f"""
                    ## ❌ Analysis Failed
                    **Error:** {error}

                    Please try a different image or check the model configuration.
                    """)
                else:
                    # ----------------- DIAGNOSIS CARD ----------------- #
                    st.subheader("📋 Diagnosis Results")

                    formatted_disease = (
                        disease.replace("___", " - ")
                               .replace("__", " - ")
                               .replace("_", " ")
                    )

                    if "healthy" in disease.lower():
                        status_emoji = "✅"
                        status_text = "Healthy Plant"
                        status_color = "#2E8B57"
                    else:
                        status_emoji = "⚠️"
                        status_text = "Needs Attention"
                        status_color = "#FFA500"

                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <div style="text-align: center; margin-bottom: 1.2rem;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                            <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem;
                                         border-radius: 15px; font-weight: 600;">
                                {status_text}
                            </span>
                        </div>
                        <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">
                            {formatted_disease}
                        </h3>
                        <div style="text-align: center;">
                            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                            <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                {confidence:.1%}
                            </h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ----------------- CONFIDENCE WARNINGS ------------- #
                    if confidence < 0.4:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Low Confidence Warning</h4>
                            <p>The model is not very confident about this diagnosis. This may be due to:</p>
                            <ul>
                                <li>Poor image quality</li>
                                <li>Unusual angle or lighting</li>
                                <li>Plant type underrepresented in training data</li>
                                <li>Multiple diseases present</li>
                            </ul>
                            <p><strong>Recommendation:</strong> Try a clearer, well-lit image focusing on the leaf.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence < 0.75:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Moderate Confidence</h4>
                            <p>This prediction has moderate confidence. You may want to:</p>
                            <ul>
                                <li>Get a second opinion from a plant expert</li>
                                <li>Upload additional images from different angles</li>
                                <li>Monitor the plant for new symptoms</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("**✅ High Confidence** – Diagnosis is likely reliable.")

                    # 🆕 SIMPLIFIED HISTORY DISPLAY (REMOVED BIAS CHECK)
                    if len(st.session_state.prediction_history) >= 2:
                        st.info(f"📊 You've analyzed {len(st.session_state.prediction_history)} images in this session")

                    # ----------------- DEBUG EXPANDER ------------------ #
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Predicted class: {disease}")
                        st.write(f"Raw confidence: {confidence}")
                        st.write(f"Model input size: {img_size}")
                        st.write(f"Available classes: {len(class_names)}")
                        st.write(f"Recent predictions: {st.session_state.prediction_history}")
                        # Use fresh image for debug predictions too
                        debug_image_data = BytesIO(st.session_state.uploaded_file_data)
                        debug_image = Image.open(debug_image_data)
                        debug_model_predictions(debug_image, model, class_names, img_size)

                    # ----------------- USER FEEDBACK ------------------- #
                    st.markdown("---")
                    st.subheader("🤔 Prediction Accuracy")
                    feedback = st.radio(
                        "Does this prediction seem correct?",
                        ["Yes, looks accurate", "No, this seems wrong", "Unsure"],
                        index=0
                    )
                    if feedback == "No, this seems wrong":
                        st.warning(
                            "Thank you for your feedback! In a future version, "
                            "we could store this to improve the model."
                        )

                    # ----------------- CARE INSTRUCTIONS --------------- #
                    st.markdown("---")
                    st.subheader("💡 Care Instructions")

                    plant_name = disease.split("_")[0] if "_" in disease else "plant"

                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = get_plant_advice(plant_name, disease)

                        if any(key in advice for key in ["OpenAI", "API key", "rate limit"]):
                            st.warning("⚠️ Using fallback care advice (AI service issue).")
                            display_fallback_advice(plant_name, disease)
                        else:
                            st.success("✅ AI-Generated Personalized Advice")
                            st.info(advice)
                    else:
                        st.warning("⚠️ Using standard care advice (AI not configured).")
                        display_fallback_advice(plant_name, disease)

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
