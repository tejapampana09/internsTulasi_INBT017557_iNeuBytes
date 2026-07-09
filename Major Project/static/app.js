// CineMatch AI App JS - Client logic

let moviesIndex = [];

// DOM Elements
const movieInput = document.getElementById('movie-input');
const submitBtn = document.getElementById('submit-btn');
const recommendForm = document.getElementById('recommend-form');
const suggestionsBox = document.getElementById('suggestions-box');
const errorBanner = document.getElementById('error-banner');
const errorMessage = document.getElementById('error-message');
const loader = document.getElementById('loader');
const resultsHeader = document.getElementById('results-header');
const queryMatchedTitle = document.getElementById('query-matched-title');
const recommendationsGrid = document.getElementById('recommendations-grid');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('status-text');

// Initialize Connection
async function initializeApp() {
    try {
        console.log("Checking server health...");
        const healthRes = await fetch('/health');
        if (!healthRes.ok) throw new Error("Health check failed");
        
        statusDot.classList.add('status-online');
        statusText.textContent = "System Online";
        
        console.log("Loading movie suggestions index...");
        const moviesRes = await fetch('/movies');
        if (!moviesRes.ok) throw new Error("Could not load movies database");
        
        moviesIndex = await moviesRes.json();
        console.log(`Loaded ${moviesIndex.length} movies for autocomplete.`);
        
        // Enable search input
        submitBtn.removeAttribute('disabled');
        movieInput.placeholder = "Type a movie title (e.g. Inception, Avatar)...";
    } catch (err) {
        console.error("Initialization error:", err);
        statusDot.classList.add('status-offline');
        statusText.textContent = "Offline / Error";
        showError("Unable to connect to the recommendation API. Please ensure the backend is running.");
    }
}

// Show/Hide Error
function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = 'flex';
    // Auto-scroll to error
    errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorBanner.style.display = 'none';
}

// Handle Auto-complete Suggestions
movieInput.addEventListener('input', () => {
    const val = movieInput.value.trim().toLowerCase();
    suggestionsBox.innerHTML = '';
    
    if (!val || val.length < 2) {
        suggestionsBox.style.display = 'none';
        return;
    }
    
    // Filter top 8 matches
    const matches = moviesIndex
        .filter(m => m.title.toLowerCase().includes(val))
        .slice(0, 8);
        
    if (matches.length === 0) {
        suggestionsBox.style.display = 'none';
        return;
    }
    
    matches.forEach(match => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.textContent = match.title;
        item.addEventListener('click', () => {
            movieInput.value = match.title;
            suggestionsBox.style.display = 'none';
            // Automatically submit
            getRecommendations(match.title);
        });
        suggestionsBox.appendChild(item);
    });
    
    suggestionsBox.style.display = 'block';
});

// Close suggestions on clicking outside
document.addEventListener('click', (e) => {
    if (e.target !== movieInput && e.target !== suggestionsBox) {
        suggestionsBox.style.display = 'none';
    }
});

// Form Submission
recommendForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const movieQuery = movieInput.value.trim();
    if (movieQuery) {
        suggestionsBox.style.display = 'none';
        getRecommendations(movieQuery);
    }
});

// Fetch Recommendations
async function getRecommendations(title) {
    hideError();
    loader.style.display = 'flex';
    resultsHeader.style.display = 'none';
    recommendationsGrid.innerHTML = '';
    
    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_title: title })
        });
        
        const data = await response.json();
        loader.style.display = 'none';
        
        if (!response.ok) {
            showError(data.message || "Failed to fetch recommendations.");
            return;
        }
        
        // Render results
        queryMatchedTitle.textContent = data.matched_title;
        resultsHeader.style.display = 'block';
        
        if (data.recommendations.length === 0) {
            recommendationsGrid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--text-secondary);">No similar movies found.</p>`;
            return;
        }
        
        renderCards(data.recommendations);
    } catch (err) {
        loader.style.display = 'none';
        console.error("Fetch error:", err);
        showError("An error occurred. Please check your network connection and try again.");
    }
}

// Parse Genres String (e.g., "['Action', 'Thriller']")
function parseGenres(genresStr) {
    try {
        if (!genresStr) return [];
        // Replace single quotes with double quotes to make it valid JSON if needed
        const clean = genresStr.replace(/'/g, '"');
        return JSON.parse(clean);
    } catch (e) {
        // Fallback for comma separated strings
        return genresStr.replace(/[\[\]']/g, '').split(',').map(s => s.trim()).filter(Boolean);
    }
}

// Render Movie Cards
function renderCards(movies) {
    movies.forEach(movie => {
        const genres = parseGenres(movie.genres);
        
        const card = document.createElement('div');
        card.className = 'glass-card movie-card';
        
        // Card Header (Title & Rating)
        const header = document.createElement('div');
        header.className = 'card-header';
        
        const h3 = document.createElement('h3');
        h3.textContent = movie.title;
        header.appendChild(h3);
        
        if (movie.vote_average > 0) {
            const rating = document.createElement('div');
            rating.className = 'rating-badge';
            rating.innerHTML = `<i class="fa-solid fa-star"></i> <span>${movie.vote_average.toFixed(1)}</span>`;
            header.appendChild(rating);
        }
        card.appendChild(header);
        
        // Genres Container
        const genresContainer = document.createElement('div');
        genresContainer.className = 'genres-container';
        
        genres.forEach(g => {
            const badge = document.createElement('span');
            badge.className = 'genre-badge';
            badge.textContent = (typeof g === 'object' && g !== null && g.name) ? g.name : g;
            genresContainer.appendChild(badge);
        });
        card.appendChild(genresContainer);
        
        // Overview Text
        const overview = document.createElement('p');
        overview.className = 'overview-text';
        overview.textContent = movie.overview || "No overview synopsis available for this movie.";
        card.appendChild(overview);
        
        // Card Actions ("More Like This" button)
        const actions = document.createElement('div');
        actions.className = 'card-actions';
        
        const actionBtn = document.createElement('button');
        actionBtn.className = 'more-like-this-btn';
        actionBtn.innerHTML = `<span>More like this</span> <i class="fa-solid fa-angle-right"></i>`;
        actionBtn.addEventListener('click', () => {
            movieInput.value = movie.title;
            getRecommendations(movie.title);
        });
        actions.appendChild(actionBtn);
        card.appendChild(actions);
        
        recommendationsGrid.appendChild(card);
    });
}

// Start Application on load
window.addEventListener('DOMContentLoaded', initializeApp);
