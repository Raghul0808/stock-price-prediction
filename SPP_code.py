<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Driven Stock Price Prediction</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
            color: #333;
        }

        .container {
            max-width: 700px; /* Increased width to accommodate the graph */
            margin: 20px auto;
            background-color: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            color: #0056b3;
            margin-bottom: 20px;
        }

        #prediction-form {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 25px;
        }

        label {
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
        }

        input[type="date"] {
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            width: 200px;
            font-size: 16px;
        }

        button {
            padding: 12px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #0056b3;
        }

        #prediction-result {
            margin-top: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background-color: #f9f9f9;
        }

        h2 {
            color: #0056b3;
            margin-top: 0;
            margin-bottom: 15px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }

        h3 {
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        p {
            line-height: 1.6;
            color: #555;
            margin-bottom: 10px;
        }

        .sentiment-label {
            font-weight: bold;
            margin-right: 5px;
        }

        .bullish {
            color: green;
            font-weight: bold;
        }

        .bearish {
            color: red;
            font-weight: bold;
        }

        .neutral {
            color: orange;
            font-weight: bold;
        }

        #company-outlook {
            list-style-type: none;
            padding: 0;
        }

        #company-outlook li {
            margin-bottom: 8px;
        }

        .profit {
            color: green;
            font-weight: bold;
        }

        .loss {
            color: red;
            font-weight: bold;
        }

        .neutral-company {
            color: gray;
            font-weight: bold;
        }

        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .analysis-table th,
        .analysis-table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }

        .analysis-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }

        .disclaimer {
            font-size: 0.9em;
            color: #777;
            margin-top: 20px;
            font-style: italic;
        }

        /* Style for the chart canvas */
        #stockChart {
            width: 100%;
            max-height: 300px;
            margin-top: 20px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
    <div class="container">
        <h1>AI Driven Stock Price Prediction</h1>
        <form id="prediction-form">
            <label for="date">Select Date:</label>
            <input type="date" id="date" name="date" required>
            <button type="button" onclick="getMarketOutlook()">Analyze Market</button>
        </form>
        <div id="prediction-result" style="display: none;">
            <h2>Market Outlook for <span id="outlook-date"></span></h2>
            <p><span class="sentiment-label">Overall Sentiment (Simulated):</span> <span id="market-sentiment"></span></p>
            <h3>Historical Stock Price (Sample)</h3>
            <canvas id="stockChart"></canvas>
            <h3>Likely Profitable/Loss-Making Companies (Simulated)</h3>
            <ul id="company-outlook"></ul>
            <h3>Simplified Stationarity Analysis (Based on Sample Stock)</h3>
            <table class="analysis-table" id="analysis-table">
            </table>
        </div>
        <p class="disclaimer">**Disclaimer:** The predictions and analysis provided here are based on very simplified simulations using limited historical data and are for demonstration purposes only. They should not be used for making actual financial decisions.</p>
    </div>

    <script>
        // Extracted data from the notebook (representing a single stock)
        const singleStockData = [
            { date: '2024-10-23', close: 1940.20 },
            { date: '2024-10-24', close: 1936.95 },
            { date: '2024-10-25', close: 1894.35 },
            { date: '2024-10-28', close: 1931.50 },
            { date: '2024-10-29', close: 1937.60 },
            { date: '2025-03-10', close: 1485.15 },
            { date: '2025-03-11', close: 1482.45 },
            { date: '2025-03-12', close: 1459.75 },
            { date: '2025-03-13', close: 1437.80 },
            { date: '2025-03-17', close: 1428.50 }
        ];

        // Simulate data for multiple companies (for demonstration)
        const companyData = {
            "Apple": [...singleStockData], // Using the same data for simplicity
            "Google": [...singleStockData].map(item => ({ ...item, close: item.close + 50 })),
            "Amazon": [...singleStockData].map(item => ({ ...item, close: item.close - 20 })),
            "Tesla": [...singleStockData].map(item => ({ ...item, close: item.close + 100 })),
            "Microsoft": [...singleStockData].map(item => ({ ...item, close: item.close - 30 }))
        };

        function calculateMean(data) {
            const sum = data.reduce((acc, val) => acc + val.close, 0);
            return sum / data.length;
        }

        function calculateStdDev(data) {
            const mean = calculateMean(data);
            const sumOfSquares = data.reduce((acc, val) => acc + Math.pow(val.close - mean, 2), 0);
            const variance = sumOfSquares / (data.length - 1);
            return Math.sqrt(variance);
        }

        function simulateSentiment(data) {
            const lastPrice = data[data.length - 1].close;
            const previousPrice = data[data.length - 2]?.close || lastPrice;

            if (lastPrice > previousPrice) {
                return "Bullish";
            } else if (lastPrice < previousPrice) {
                return "Bearish";
            } else {
                return "Neutral";
            }
        }

        function simulateCompanyProfitability(companyName, data) {
            const lastPrice = data[data.length - 1].close;
            const firstPrice = data[0].close;

            if (lastPrice > firstPrice) {
                return "Likely Profitable";
            } else if (lastPrice < firstPrice) {
                return "Likely Loss-Making";
            } else {
                return "Neutral";
            }
        }

        function simplifiedDickeyFuller(data) {
            const closePrices = data.map(item => item.close);
            const diff = closePrices.slice(1).map((value, index) => value - closePrices[index]);
            const diffMean = calculateMean(diff.map(c => ({ close: c })));

            let isStationary = "Likely Not Stationary (Simplified)";
            if (Math.abs(diffMean) < 1) {
                isStationary = "Likely Stationary (Simplified)";
            }

            return {
                "Simplified Stationarity": isStationary,
                "First Difference Mean": Math.round(diffMean * 100) / 100
            };
        }

        function getMarketOutlook() {
            const date = document.getElementById('date').value;
            const outlookDateElement = document.getElementById('outlook-date');
            const marketSentimentElement = document.getElementById('market-sentiment');
            const companyOutlookList = document.getElementById('company-outlook');
            const analysisTable = document.getElementById('analysis-table');
            const predictionResultDiv = document.getElementById('prediction-result');
            const stockChartCanvas = document.getElementById('stockChart').getContext('2d');

            outlookDateElement.textContent = date;

            const sentiment = simulateSentiment(singleStockData);
            marketSentimentElement.textContent = sentiment;
            marketSentimentElement.className = sentiment.toLowerCase();

            companyOutlookList.innerHTML = '';
            for (const company in companyData) {
                const profitability = simulateCompanyProfitability(company, companyData[company]);
                const listItem = document.createElement('li');
                listItem.textContent = `${company}: `;
                const span = document.createElement('span');
                span.textContent = profitability;
                span.className = profitability.toLowerCase().replace('likely ', '').replace('-', '');
                listItem.appendChild(span);
                companyOutlookList.appendChild(listItem);
            }

            const dfResults = simplifiedDickeyFuller(singleStockData);
            let tableHTML = '<tr><th>Metric</th><th>Value</th></tr>';
            for (const key in dfResults) {
                tableHTML += `<tr><td>${key}</td><td>${dfResults[key]}</td></tr>`;
            }
            analysisTable.innerHTML = tableHTML;

            // --- Create the Line Graph ---
            const labels = singleStockData.map(item => item.date);
            const prices = singleStockData.map(item => item.close);

            new Chart(stockChartCanvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Historical Stock Price (Sample)',
                        data: prices,
                        borderColor: 'blue',
                        borderWidth: 2,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Closing Price'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        }
                    }
                }
            });

            predictionResultDiv.style.display = 'block'; // Show the results
        }
    </script>
</body>

</html>
