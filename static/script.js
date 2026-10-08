// Keyword Extraction Web App - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('inputText');
    const extractBtn = document.getElementById('extractBtn');
    const clearBtn = document.getElementById('clearBtn');
    const errorMessage = document.getElementById('errorMessage');
    const resultsSection = document.getElementById('resultsSection');
    const btnText = extractBtn.querySelector('.btn-text');
    const spinner = extractBtn.querySelector('.spinner');

    // Extract button click handler
    extractBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();
        
        // Validation
        if (!text) {
            showError('Please enter some text.');
            return;
        }
        
        if (text.length < 10) {
            showError('Text too short. Please enter at least 10 characters.');
            return;
        }
        
        // Clear previous error
        hideError();
        
        // Show loading state
        setLoading(true);
        
        try {
            const response = await fetch('http://127.0.0.1:5000/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to extract keywords');
            }
            
            // Display results
            displayResults(data.results);
            resultsSection.classList.remove('hidden');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            showError(error.message);
        } finally {
            setLoading(false);
        }
    });

    // Clear button click handler
    clearBtn.addEventListener('click', function() {
        inputText.value = '';
        hideError();
        resultsSection.classList.add('hidden');
        inputText.focus();
    });

    // Set loading state
    function setLoading(loading) {
        extractBtn.disabled = loading;
        if (loading) {
            btnText.textContent = 'Processing...';
            spinner.classList.remove('hidden');
        } else {
            btnText.textContent = 'Extract Keywords';
            spinner.classList.add('hidden');
        }
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    // Hide error message
    function hideError() {
        errorMessage.classList.add('hidden');
    }

    // Display results
    function displayResults(results) {
        // RAKE Results
        displayRakeResults(results.rake);
        
        // TF-IDF Results
        displayTfidfResults(results.tfidf);
        
        // KeyBERT Results
        displayKeybertResults(results.keybert);
    }

    // Display RAKE results with all calculation steps
    function displayRakeResults(rakeData) {
        const steps = rakeData.calculation_details.steps;
        
        // Step 1: Stopwords
        const step1Div = document.getElementById('rakeStep1');
        step1Div.innerHTML = `
            <p><strong>Stopwords found:</strong> ${steps.step1_stopwords.stopwords.join(', ')}</p>
            <p><strong>Total unique stopwords:</strong> ${steps.step1_stopwords.count}</p>
        `;
        
        // Step 2: Phrases
        const step2List = document.getElementById('rakeStep2');
        step2List.innerHTML = '';
        steps.step2_phrases.forEach((phrase, index) => {
            if (phrase.trim()) {
                const li = document.createElement('li');
                li.textContent = `"${phrase}"`;
                step2List.appendChild(li);
            }
        });
        
        // Step 3: Word Frequencies
        const step3Table = document.getElementById('rakeStep3');
        step3Table.innerHTML = '';
        steps.step3_word_frequencies.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.word}</td><td>${item.frequency}</td>`;
            step3Table.appendChild(row);
        });
        
        // Step 4: Word Degrees
        const step4Table = document.getElementById('rakeStep4');
        step4Table.innerHTML = '';
        steps.step4_word_degrees.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.word}</td><td>${item.degree}</td><td>${item.explanation}</td>`;
            step4Table.appendChild(row);
        });
        
        // Step 5: Word Scores
        const step5Table = document.getElementById('rakeStep5');
        step5Table.innerHTML = '';
        steps.step5_word_scores.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.word}</td><td>${item.calculation}</td><td>${parseFloat(item.score).toFixed(3)}</td>`;
            step5Table.appendChild(row);
        });
        
        // Step 6: Phrase Scores
        const step6Div = document.getElementById('rakeStep6');
        step6Div.innerHTML = '';
        steps.step6_phrase_scores.forEach(item => {
            const div = document.createElement('div');
            div.className = 'phrase-calc-item';
            div.innerHTML = `
                <span class="phrase-name">"${item.phrase}"</span>
                <span class="phrase-calc"> = ${item.calculation} = </span>
                <span class="phrase-total">${parseFloat(item.total).toFixed(3)}</span>
            `;
            step6Div.appendChild(div);
        });
        
        // Keywords table
        const rakeTable = document.getElementById('rakeKeywords');
        rakeTable.innerHTML = '';
        rakeData.keywords.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.keyword}</td>
                <td>${parseFloat(item.score).toFixed(3)}</td>
            `;
            rakeTable.appendChild(row);
        });
    }

    // Display TF-IDF results with all calculation steps
    function displayTfidfResults(tfidfData) {
        const steps = tfidfData.calculation_details.steps;
        const strategy = tfidfData.calculation_details.strategy || '';
        const totalDocs = tfidfData.calculation_details.total_documents || steps.step1_sentences.length;

        // Step 1: Sentences / Documents
        const step1Div = document.getElementById('tfidfStep1');
        step1Div.innerHTML = '';

        // Strategy banner
        if (strategy) {
            const banner = document.createElement('div');
            banner.className = 'strategy-banner';
            banner.innerHTML = `
                <span class="strategy-label">🔍 Auto-detected:</span>
                <span class="strategy-name">${strategy}</span>
                <span class="doc-count-badge">${totalDocs} document${totalDocs !== 1 ? 's' : ''} found</span>
            `;
            step1Div.appendChild(banner);
        }

        steps.step1_sentences.forEach((sentence, index) => {
            const p = document.createElement('p');
            p.innerHTML = `<strong>Document ${index + 1}:</strong> "${sentence}"`;
            step1Div.appendChild(p);
        });

        // Step 2: Processed Sentences
        const step2Div = document.getElementById('tfidfStep2');
        step2Div.innerHTML = '';
        steps.step2_processed_sentences.forEach((sentence, index) => {
            const p = document.createElement('p');
            p.innerHTML = `<strong>Document ${index + 1}:</strong> [${sentence}]`;
            step2Div.appendChild(p);
        });

        // Step 3: Term Frequencies per Document
        const step3Container = document.getElementById('tfidfStep3');
        step3Container.innerHTML = '';
        steps.step3_per_sentence_tf.forEach((tf_data, index) => {
            const h5 = document.createElement('h5');
            h5.textContent = `Document ${index + 1}`;
            h5.style.marginTop = '15px';
            step3Container.appendChild(h5);

            const table = document.createElement('table');
            table.className = 'calc-table';
            table.innerHTML = `
                <thead><tr><th>Word</th><th>Count</th><th>Calculation</th><th>TF</th></tr></thead>
                <tbody></tbody>
            `;
            const tbody = table.querySelector('tbody');

            tf_data.words.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${item.word}</td><td>${item.count}</td><td>${item.calculation}</td><td>${parseFloat(item.tf).toFixed(3)}</td>`;
                tbody.appendChild(row);
            });

            step3Container.appendChild(table);
        });

        // Step 4: IDF Calculation
        document.getElementById('tfidfCorpusSize').textContent = totalDocs;
        const step4Table = document.getElementById('tfidfStep4').parentNode.querySelector('tbody');
        step4Table.innerHTML = '';
        steps.step4_idf_calculation.idf_values.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.word}</td><td>${item.df}</td><td>${item.calculation}</td><td>${parseFloat(item.idf).toFixed(3)}</td>`;
            step4Table.appendChild(row);
        });

        // Step 5: TF-IDF Scores per Document
        const step5Table = document.getElementById('tfidfStep5');
        // step5Table is a tbody — clear it and instead use its parent section
        const step5Container = step5Table.closest('.calculation-step') || step5Table.parentNode;
        // We'll render into the tbody using a collapsible per-doc approach
        step5Table.innerHTML = '';

        const tfidfDocs = steps.step5_tfidf_docs || [];
        const finalScores = steps.step5_final_scores || [];

        if (tfidfDocs.length > 0) {
            // Per-document tables — rendered above the aggregated table
            const perDocSection = document.createElement('div');
            perDocSection.style.marginBottom = '16px';

            tfidfDocs.forEach(docData => {
                const docTitle = document.createElement('h5');
                docTitle.textContent = `Document ${docData.doc_index}: "${String(docData.doc_text).substring(0, 60)}${docData.doc_text.length > 60 ? '…' : ''}"`;
                docTitle.style.marginTop = '14px';
                docTitle.style.marginBottom = '6px';
                docTitle.style.color = '#444';
                perDocSection.appendChild(docTitle);

                if (!docData.scores || docData.scores.length === 0) {
                    const empty = document.createElement('p');
                    empty.textContent = 'No content words found in this document.';
                    empty.style.color = '#999';
                    perDocSection.appendChild(empty);
                    return;
                }

                const tbl = document.createElement('table');
                tbl.className = 'calc-table';
                tbl.innerHTML = `
                    <thead><tr><th>Word</th><th>TF</th><th>IDF</th><th>Score</th><th>Uniqueness</th></tr></thead>
                    <tbody></tbody>
                `;
                const tb = tbl.querySelector('tbody');
                docData.scores.forEach(item => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${item.word}</td>
                        <td>${parseFloat(item.tf).toFixed(3)}</td>
                        <td>${parseFloat(item.idf).toFixed(3)}</td>
                        <td>${parseFloat(item.score).toFixed(3)}</td>
                        <td><span class="uniqueness-badge ${item.uniqueness.toLowerCase().includes('unique') ? 'unique' : 'common'}">${item.uniqueness}</span></td>
                    `;
                    tb.appendChild(row);
                });
                perDocSection.appendChild(tbl);
            });

            // Insert per-doc tables before step5Table's parent table
            const parentTable = step5Table.closest('table');
            if (parentTable) {
                parentTable.parentNode.insertBefore(perDocSection, parentTable);
            }
        } else {
            // Fallback: aggregated view
            finalScores.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.word}</td>
                    <td>${parseFloat(item.tf).toFixed(3)}</td>
                    <td>${parseFloat(item.idf).toFixed(3)}</td>
                    <td>${parseFloat(item.score).toFixed(3)}</td>
                    <td><span class="uniqueness-badge ${item.uniqueness.toLowerCase().includes('unique') ? 'unique' : 'common'}">${item.uniqueness}</span></td>
                `;
                step5Table.appendChild(row);
            });
        }


        // Keywords table
        const tfidfTable = document.getElementById('tfidfKeywords');
        tfidfTable.innerHTML = '';
        tfidfData.keywords.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.keyword}</td>
                <td>${parseFloat(item.score).toFixed(3)}</td>
            `;
            tfidfTable.appendChild(row);
        });
    }

    // Display KeyBERT results with all calculation steps
    function displayKeybertResults(keybertData) {
        const steps = keybertData.calculation_details.steps;
        
        // Step 1: Document Embedding
        document.getElementById('keybertStep1').innerHTML = `<p>${steps.step1_document_embedding}</p>`;
        
        // Step 2: Candidate Keywords
        document.getElementById('keybertStep2').innerHTML = `<p>${steps.step2_candidate_keywords}</p>`;
        
        // Step 3: Keyword Embeddings
        document.getElementById('keybertStep3').innerHTML = `<p>${steps.step3_keyword_embeddings}</p>`;
        
        // Step 4: Cosine Similarity Explanation
        document.getElementById('keybertStep4Desc').innerHTML = `<p>${steps.step4_cosine_similarity}</p>`;
        
        // Similarity scores table
        const step4Table = document.getElementById('keybertStep4');
        step4Table.innerHTML = '';
        steps.similarity_scores.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.keyword}</td><td>${parseFloat(item.score).toFixed(3)}</td>`;
            step4Table.appendChild(row);
        });
        
        // Keywords table
        const keybertTable = document.getElementById('keybertKeywords');
        keybertTable.innerHTML = '';
        keybertData.keywords.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.keyword}</td>
                <td>${parseFloat(item.score).toFixed(3)}</td>
            `;
            keybertTable.appendChild(row);
        });
    }
});
