// Enhanced Progressive Web App for Intelligent Pesticide Spraying System
class SmartFarmApp {
    constructor() {
        this.currentLanguage = 'en';
        this.isAuthenticated = false;
        this.userData = null;
        
        // MQTT Configuration - DATA ANALYSIS ONLY
        this.mqttConfig = {
            host: '10.19.174.75',  // Your Pi IP
            port: 9001,            // WebSocket port on Pi
            topics: {
                sprayingLogs: 'agrinet/spraying/logs',    // Spraying activity logs
                pestDetection: 'agrinet/pest/detected',   // Pest ID + name only
                diseaseDetection: 'agrinet/disease/detected' // Disease ID + name only
            }
        };

        // Data storage for analysis
        this.sprayingData = {
            actionableLogs: [],   // State="actionable" - actual spraying
            unknownLogs: [],      // State="unknown" - safety stops
            dailyUsage: {},       // Usage by date
            weeklyUsage: {},      // Usage by week
            totalUsage: {
                insecticide: 0,   // Channel A total
                fungicide: 0      // Channel B total
            }
        };

        // Pest database (ID -> Name mapping)
        this.pestDatabase = {
            1: "Rice Leaf Roller", 2: "Rice Leaf Caterpillar", 3: "Paddy Stem Maggot",
            4: "Asiatic Rice Borer", 5: "Yellow Rice Borer", 6: "Rice Gall Midge",
            7: "Rice Stemfly", 8: "Brown Plant Hopper", 9: "White Backed Plant Hopper",
            10: "Small Brown Plant Hopper", 11: "Rice Water Weevil", 12: "Rice Leafhopper",
            13: "Grain Spreader Thrips", 14: "Rice Shell Pest", 15: "Grub",
            16: "Mole Cricket", 17: "Wireworm", 18: "White Margined Moth",
            19: "Black Cutworm", 20: "Large Cutworm", 21: "Yellow Cutworm",
            22: "Red Spider", 23: "Corn Borer", 24: "Army Worm", 25: "Aphids",
            26: "Potosiabre Vitarsis", 27: "Peach Borer", 28: "English Grain Aphid",
            29: "Green Bug", 30: "Bird Cherry-Oat Aphid", 31: "Wheat Blossom Midge",
            32: "Penthaleus Major", 33: "Longlegged Spider Mite", 34: "Wheat Phloeothrips",
            35: "Wheat Sawfly", 36: "Cerodonta Denticornis", 37: "Beet Fly",
            38: "Flea Beetle", 39: "Cabbage Army Worm", 40: "Beet Army Worm",
            41: "Beet Spot Flies", 42: "Meadow Moth", 43: "Beet Weevil",
            44: "Sericaorient Alismots Chulsky", 45: "Alfalfa Weevil", 46: "Flax Budworm",
            47: "Alfalfa Plant Bug", 48: "Tarnished Plant Bug", 49: "Locustoidea",
            50: "Lytta Polita", 51: "Legume Blister Beetle", 52: "Blister Beetle",
            53: "Therioaphis Maculata Buckton", 54: "Odontothrips Loti", 55: "Thrips",
            56: "Alfalfa Seed Chalcid", 57: "Pieris Canidia", 58: "Apolygus Lucorum",
            59: "Limacodidae", 60: "Viteus Vitifoliae", 61: "Colomerus Vitis",
            62: "Brevipoalpus Lewisi McGregor", 63: "Oides Decempunctata", 64: "Polyphagotars Onemus Latus",
            65: "Pseudococcus Comstocki Kuwana", 66: "Parathrene Regalis", 67: "Ampelophaga",
            68: "Lycorma Delicatula", 69: "Xylotrechus", 70: "Cicadella Viridis",
            71: "Miridae", 72: "Trialeurodes Vaporariorum", 73: "Erythroneura Apicalis",
            74: "Papilio Xuthus", 75: "Panonchus Citri McGregor", 76: "Phyllocoptes Oleiverus Ashmead",
            77: "Icerya Purchasi Maskell", 78: "Unaspis Yanonensis", 79: "Ceroplastes Rubens",
            80: "Chrysomphalus Aonidum", 81: "Parlatoria Zizyphus Lucus", 82: "Nipaecoccus Vastalor",
            83: "Aleurocanthus Spiniferus", 84: "Tetradacus C Bactrocera Minax", 85: "Dacus Dorsalis(Hendel)",
            86: "Bactrocera Tsuneonis", 87: "Prodenia Litura", 88: "Adristyrannus",
            89: "Phyllocnistis Citrella Stainton", 90: "Toxoptera Citricidus", 91: "Toxoptera Aurantii",
            92: "Aphis Citricola Vander Goot", 93: "Scirtothrips Dorsalis Hood", 94: "Dasineura Sp",
            95: "Lawana Imitata Melichar", 96: "Salurnis Marginella Guerr", 97: "Deporaus Marginatus Pascoe",
            98: "Chlumetia Transversa", 99: "Mango Flat Beak Leafhopper", 100: "Rhytidodera Bowrinii White",
            101: "Sternochetus Frigidus", 102: "Cicadellidae"
        };

        // Disease database (ID -> Name mapping)
        this.diseaseDatabase = {
            1: "Apple - Scab", 2: "Apple - Black Rot", 3: "Apple - Cedar Rust", 4: "Apple - Healthy",
            5: "Blueberry - Healthy", 6: "Cherry - Healthy", 7: "Cherry - Powdery Mildew",
            8: "Corn - Gray Leaf Spot", 9: "Corn - Common Rust", 10: "Corn - Healthy",
            11: "Corn - Northern Leaf Blight", 12: "Grape - Black Rot", 13: "Grape - Black Measles",
            14: "Grape - Leaf Blight", 15: "Grape - Healthy", 16: "Orange - Huanglongbing",
            17: "Peach - Bacterial Spot", 18: "Peach - Healthy", 19: "Bell Pepper - Bacterial Spot",
            20: "Bell Pepper - Healthy", 21: "Potato - Early Blight", 22: "Potato - Healthy",
            23: "Potato - Late Blight", 24: "Raspberry - Healthy", 25: "Soybean - Healthy",
            26: "Squash - Powdery Mildew", 27: "Strawberry - Healthy", 28: "Strawberry - Leaf Scorch",
            29: "Tomato - Bacterial Spot", 30: "Tomato - Early Blight", 31: "Tomato - Late Blight",
            32: "Tomato - Leaf Mold", 33: "Tomato - Septoria Leaf Spot", 34: "Tomato - Two Spotted Spider Mite",
            35: "Tomato - Target Spot", 36: "Tomato - Mosaic Virus", 37: "Tomato - Yellow Leaf Curl Virus",
            38: "Tomato - Healthy", 39: "Background"
        };

        
        this.mqttClient = null;
        this.isConnectedToMQTT = false;

        this.weatherConfig = {
            apiKey: 'c9c6b0ea3596b06800c7942195c4ebe0',  
            location: {
                lat: 11.6643,   
                lon: 78.1460
            },
            updateInterval: 300000, 
            endpoints: {
                current: 'https://api.openweathermap.org/data/2.5/weather',
                forecast: 'https://api.openweathermap.org/data/2.5/forecast'
            }
        };

        this.weatherUpdateTimer = null;

        
        // Your existing mock data (unchanged)
        this.appData = {
            farmHealth: {
                overallStatus: "attention",
                activeDiseases: 3,
                activePests: 2,
                lastUpdated: "2025-09-28T14:30:00Z",
                treatmentEffectiveness: 78
            },
            diseases: [
                {
                    id: 12,
                    name: "Early Blight",
                    nameTamil: "முன்கூட்டிய தீங்கு",
                    severity: "medium",
                    location: "Field Alpha - Section B",
                    detectedAt: "2025-09-28T10:15:00Z",
                    confidenceLevel: 87,
                    recommendation: "Apply Mancozeb 75% WP @ 2kg/ha"
                },
                {
                    id: 8,
                    name: "Bacterial Wilt",
                    nameTamil: "பாக்டீரியா வாடல்",
                    severity: "high",
                    location: "Field Alpha - Section A",
                    detectedAt: "2025-09-28T08:45:00Z",
                    confidenceLevel: 92,
                    recommendation: "Remove affected plants, apply Streptomycin"
                },
                {
                    id: 25,
                    name: "Powdery Mildew",
                    nameTamil: "பவுடரி பூஞ்சை",
                    severity: "low",
                    location: "Field Alpha - Section C",
                    detectedAt: "2025-09-27T16:30:00Z",
                    confidenceLevel: 76,
                    recommendation: "Apply Sulfur 80% WP @ 3kg/ha"
                }
            ],
            pests: [
                {
                    id: 23,
                    name: "Aphids",
                    nameTamil: "அழிகு பூச்சி",
                    severity: "medium",
                    location: "Field Alpha - Section B",
                    population: "15-20 per plant",
                    detectedAt: "2025-09-28T11:20:00Z",
                    confidenceLevel: 89,
                    economicThreshold: "approaching",
                    recommendation: "Apply Imidacloprid 17.8% SL @ 0.3ml/L"
                },
                {
                    id: 67,
                    name: "Whitefly",
                    nameTamil: "வெள்ளை ஈ",
                    severity: "high",
                    location: "Field Alpha - Section A",
                    population: "50+ per plant",
                    detectedAt: "2025-09-28T09:10:00Z",
                    confidenceLevel: 94,
                    economicThreshold: "exceeded",
                    recommendation: "Apply Thiamethoxam 25% WG @ 0.2g/L"
                }
            ],
            tanks: {
                insecticide: {
                    currentLevel: 65,
                    capacity: 200,
                    currentLiters: 130,
                    activeIngredient: "Imidacloprid 17.8%",
                    lastRefill: "2025-09-26T08:00:00Z",
                    usageToday: 12.5,
                    usageWeek: 45.2,
                    status: "normal"
                },
                fungicide: {
                    currentLevel: 45,
                    capacity: 150,
                    currentLiters: 67.5,
                    activeIngredient: "Mancozeb 75%",
                    lastRefill: "2025-09-25T10:30:00Z",
                    expiryDate: "2025-12-15",
                    usageToday: 8.3,
                    usageWeek: 32.7,
                    status: "attention"
                },
                water: {
                    currentLevel: 82,
                    capacity: 500,
                    currentLiters: 410,
                    quality: "good",
                    filterStatus: "clean",
                    flowRate: 15.2,
                    lastRefill: "2025-09-28T06:00:00Z",
                    status: "normal"
                }
            },
            weather: {
                location: "Salem, Tamil Nadu",
                currentTemp: 40,
                condition: "partly_cloudy",
                humidity: 72,
                windSpeed: 80,
                sprayConditions: "favorable",
                diseaseRisk: "medium",
                alerts: [
                    {
                        type: "wind_warning",
                        message: "Wind speed may increase to 15+ km/h after 4 PM",
                        severity: "medium",
                        validUntil: "2025-09-28T16:00:00Z"
                    }
                ],
                forecast: [
                    {"date": "2025-09-28", "high": 32, "low": 24, "condition": "sunny", "humidity": 65, "windSpeed": 12},
                    {"date": "2025-09-29", "high": 30, "low": 23, "condition": "cloudy", "humidity": 78, "windSpeed": 8},
                    {"date": "2025-09-30", "high": 29, "low": 22, "condition": "light_rain", "humidity": 85, "windSpeed": 6}
                ]
            },
            recommendations: [
                {
                    type: "immediate_action",
                    priority: "high",
                    title: "Whitefly Treatment Required",
                    description: "Population exceeded economic threshold in Section A",
                    treatment: "Apply Thiamethoxam 25% WG @ 0.2g/L",
                    dosage: "40g for 200L water tank",
                    timing: "Early morning (6-8 AM) or evening (4-6 PM)",
                    safetyPeriod: "7 days"
                },
                {
                    type: "preventive",
                    priority: "medium", 
                    title: "Early Blight Prevention",
                    description: "Weather conditions favorable for disease spread",
                    treatment: "Apply Mancozeb 75% WP",
                    dosage: "2kg per hectare",
                    timing: "Before evening dew formation",
                    safetyPeriod: "14 days"
                }
            ]
        };
        
        this.translations = {
            en: {
                appTitle: "Smart Farm Monitor",
                emergency: "Emergency",
                farmHealthOverview: "Farm Health Overview",
                attention: "Attention Required",
                activeDiseases: "Active Diseases",
                activePests: "Active Pests",
                treatmentEffectiveness: "Treatment Effectiveness",
                tankStatus: "Tank Status",
                refillAll: "Refill All",
                insecticide: "Insecticide",
                fungicide: "Fungicide",
                water: "Water",
                capacity: "Capacity",
                activeIngredient: "Active",
                usageToday: "Today",
                quality: "Quality",
                flowRate: "Flow Rate",
                diseases: "Diseases",
                pests: "Pests",
                active: "Active",
                weather: "Weather",
                sprayConditions: "Spray Conditions",
                diseaseRisk: "Disease Risk",
                recommendations: "Recommendations",
                phoneNumber: "Phone Number",
                enterOTP: "Enter OTP",
                verify: "Verify",
                severity: "Severity",
                location: "Location",
                treatment: "Treatment",
                dosage: "Dosage",
                immediate: "Immediate Action",
                preventive: "Preventive"
            },
            ta: {
                appTitle: "ஸ்மார்ட் பண்ணை கண்காணிப்பு",
                emergency: "அவசரம்",
                farmHealthOverview: "பண்ணை நலம் மேலோட்டம்",
                attention: "கவனம் தேவை",
                activeDiseases: "செயல்படும் நோய்கள்",
                activePests: "செயல்படும் பூச்சிகள்",
                treatmentEffectiveness: "சிகிச்சை செயல்திறன்",
                tankStatus: "தொட்டி நிலை",
                refillAll: "அனைத்தையும் நிரப்பு",
                insecticide: "பூச்சிக்கொல்லி",
                fungicide: "பூஞ்சைக்கொல்லி",
                water: "நீர்",
                capacity: "திறன்",
                activeIngredient: "செயல்படும்",
                usageToday: "இன்று",
                quality: "தரம்",
                flowRate: "ஓட்ட விகிதம்",
                diseases: "நோய்கள்",
                pests: "பூச்சிகள்",
                active: "செயல்படும்",
                weather: "வானிலை",
                sprayConditions: "தெளிக்கும் நிலைமை",
                diseaseRisk: "நோய் ஆபத்து",
                recommendations: "பரிந்துரைகள்",
                phoneNumber: "தொலைபேசி எண்",
                enterOTP: "OTP உள்ளிடுக",
                verify: "சரிபார்க்கவும்",
                severity: "தீவிரம்",
                location: "இடம்",
                treatment: "சிகிச்சை",
                dosage: "அளவு",
                immediate: "உடனடி நடவடிக்கை",
                preventive: "தடுப்பு"
            }
        };

        this.init();
    }

    init() {
        this.registerServiceWorker();
        this.setupEventListeners();
        this.showSplashScreen();
        this.startDataSimulation();
        this.connectToMQTT(); // NEW: Connect to MQTT
    }

    // MQTT DATA ANALYSIS FUNCTIONS - REPLACE ALL MQTT FUNCTIONS

    async connectToMQTT() {

        try {
            const brokerUrl = `ws://${this.mqttConfig.host}:${this.mqttConfig.port}`;
            console.log('📊 Connecting to farm data analysis system:', brokerUrl);
            
            this.mqttClient = mqtt.connect(brokerUrl, {
                clientId: 'smart-farm-analyzer-' + Math.random().toString(16).substr(2, 8),
                clean: true,
                connectTimeout: 4000,
                reconnectPeriod: 1000,
                keepalive: 60
            });

            this.mqttClient.on('connect', () => {
                console.log('✅ Connected to farm data analysis system');
                this.isConnectedToMQTT = true;
                this.updateConnectionStatus();
                this.subscribeToDataStreams();
                this.showSuccess('Connected to farm data analysis system');
            });

            this.mqttClient.on('message', (topic, message) => {
                this.handleFarmDataStream(topic, message);
            });

            this.mqttClient.on('error', (error) => {
                console.error('❌ Data Analysis System Error:', error);
                this.isConnectedToMQTT = false;
                this.updateConnectionStatus();
            });

            this.mqttClient.on('close', () => {
                console.log('⚠️ Disconnected from farm data analysis system');
                this.isConnectedToMQTT = false;
                this.updateConnectionStatus();
            });

        } catch (error) {
            console.error('Failed to connect to data analysis system:', error);
            this.showError('Cannot connect to farm data analysis system');
        }
    }

    subscribeToDataStreams() {
        const topics = Object.values(this.mqttConfig.topics);
        
        topics.forEach(topic => {
            this.mqttClient.subscribe(topic, (error) => {
                if (error) {
                    console.error(`Failed to subscribe to ${topic}:`, error);
                } else {
                    console.log(`📡 Analyzing data from: ${topic}`);
                }
            });
        });
        
        console.log('📊 Subscribed to all farm data streams');
    }

    handleFarmDataStream(topic, message) {
        try {
            const data = JSON.parse(message.toString());
            console.log(`📨 Farm Data Stream [${topic}]:`, data);

            switch (topic) {
                case this.mqttConfig.topics.sprayingLogs:
                    this.analyzeSprayingLog(data);
                    break;
                    
                case this.mqttConfig.topics.pestDetection:
                    this.recordPestDetection(data);
                    break;
                    
                case this.mqttConfig.topics.diseaseDetection:
                    this.recordDiseaseDetection(data);
                    break;
            }
            
        } catch (error) {
            console.error('Error parsing farm data stream:', error);
        }
    }

    analyzeSprayingLog(data) {
        /*
        Expected Pi spraying log format:
        {
            "seq": 1234567890,
            "state": "actionable", // or "unknown"
            "channel": "A",        // A=insecticide, B=fungicide
            "doseA_ml": 250,       // ml of insecticide used
            "doseB_ml": 0,         // ml of fungicide used  
            "ttl_ms": 300,
            "timestamp": "2025-09-28T10:15:00Z",
            "duration": 30,        // seconds valve was open
            "location": "Field A - Section 2"
        }
        */
        
        console.log('🚿 Analyzing spraying log:', data);
        
        const logEntry = {
            seq: data.seq || 0,
            state: data.state || 'unknown',
            channel: data.channel || 'none',
            doseA_ml: data.doseA_ml || 0,
            doseB_ml: data.doseB_ml || 0,
            timestamp: data.timestamp || new Date().toISOString(),
            duration: data.duration || 0,
            location: data.location || 'Unknown location'
        };
        
        // Separate actionable vs unknown logs
        if (logEntry.state === 'actionable') {
            this.sprayingData.actionableLogs.unshift(logEntry);
            if (this.sprayingData.actionableLogs.length > 100) {
                this.sprayingData.actionableLogs.pop();
            }
            
            // Analyze usage patterns
            this.analyzeUsagePatterns(logEntry);
            
        } else {
            this.sprayingData.unknownLogs.unshift(logEntry);
            if (this.sprayingData.unknownLogs.length > 50) {
                this.sprayingData.unknownLogs.pop();
            }
        }
        
        // Update dashboard with analysis
        this.updateUsageAnalysis();
        this.updateSprayingLogs();
        
        // Show usage notification
        if (logEntry.state === 'actionable' && (logEntry.doseA_ml > 0 || logEntry.doseB_ml > 0)) {
            const substance = logEntry.channel === 'A' ? 'insecticide' : 'fungicide';
            const amount = logEntry.channel === 'A' ? logEntry.doseA_ml : logEntry.doseB_ml;
            this.showNotification(`🚿 ${substance}: ${amount}ml used in ${logEntry.location}`, 'info');
        }
    }

    analyzeUsagePatterns(logEntry) {
        const date = logEntry.timestamp.split('T')[0]; // Get YYYY-MM-DD
        const week = this.getWeekNumber(new Date(logEntry.timestamp));
        
        // Initialize daily usage if not exists
        if (!this.sprayingData.dailyUsage[date]) {
            this.sprayingData.dailyUsage[date] = {
                insecticide: 0,
                fungicide: 0,
                applications: 0,
                locations: new Set()
            };
        }
        
        // Initialize weekly usage if not exists
        if (!this.sprayingData.weeklyUsage[week]) {
            this.sprayingData.weeklyUsage[week] = {
                insecticide: 0,
                fungicide: 0,
                applications: 0,
                locations: new Set()
            };
        }
        
        // Update usage data
        const dailyData = this.sprayingData.dailyUsage[date];
        const weeklyData = this.sprayingData.weeklyUsage[week];
        
        if (logEntry.channel === 'A' && logEntry.doseA_ml > 0) {
            // Insecticide usage
            dailyData.insecticide += logEntry.doseA_ml;
            weeklyData.insecticide += logEntry.doseA_ml;
            this.sprayingData.totalUsage.insecticide += logEntry.doseA_ml;
        }
        
        if (logEntry.channel === 'B' && logEntry.doseB_ml > 0) {
            // Fungicide usage
            dailyData.fungicide += logEntry.doseB_ml;
            weeklyData.fungicide += logEntry.doseB_ml;
            this.sprayingData.totalUsage.fungicide += logEntry.doseB_ml;
        }
        
        // Track applications and locations
        dailyData.applications++;
        weeklyData.applications++;
        dailyData.locations.add(logEntry.location);
        weeklyData.locations.add(logEntry.location);
        
        console.log('📈 Usage analysis updated:', {
            date: date,
            daily: dailyData,
            total: this.sprayingData.totalUsage
        });
    }

    recordPestDetection(data) {
        /*
        Expected Pi pest detection format:
        {
            "pestId": 25,          // ID from 1-102
            "timestamp": "2025-09-28T10:15:00Z",
            "location": "Field A - Section 2"
        }
        */
        
        const pestName = this.pestDatabase[data.pestId] || `Unknown Pest (ID: ${data.pestId})`;
        
        const pestRecord = {
            id: data.pestId,
            name: pestName,
            timestamp: data.timestamp || new Date().toISOString(),
            location: data.location || 'Unknown location'
        };
        
        // Add to pest list (keep last 20)
        this.appData.pests.unshift(pestRecord);
        if (this.appData.pests.length > 20) {
            this.appData.pests.pop();
        }
        
        this.appData.farmHealth.activePests = this.appData.pests.length;
        
        // Update displays
        this.updatePestMonitoring();
        this.updateFarmHealth();
        
        console.log('🐛 Pest recorded:', pestRecord);
        this.showNotification(`🐛 Pest detected: ${pestName}`, 'warning');
    }

    recordDiseaseDetection(data) {
        /*
        Expected Pi disease detection format:
        {
            "diseaseId": 21,       // ID from 1-39
            "timestamp": "2025-09-28T10:15:00Z", 
            "location": "Field A - Section 2"
        }
        */
        
        const diseaseName = this.diseaseDatabase[data.diseaseId] || `Unknown Disease (ID: ${data.diseaseId})`;
        
        const diseaseRecord = {
            id: data.diseaseId,
            name: diseaseName,
            timestamp: data.timestamp || new Date().toISOString(),
            location: data.location || 'Unknown location'
        };
        
        // Add to disease list (keep last 20)
        this.appData.diseases.unshift(diseaseRecord);
        if (this.appData.diseases.length > 20) {
            this.appData.diseases.pop();
        }
        
        this.appData.farmHealth.activeDiseases = this.appData.diseases.length;
        
        // Update displays
        this.updateDiseaseDetection();
        this.updateFarmHealth();
        
        console.log('🦠 Disease recorded:', diseaseRecord);
        this.showNotification(`🦠 Disease detected: ${diseaseName}`, 'warning');
    }

    // DATA VISUALIZATION FUNCTIONS

    updateUsageAnalysis() {
        // Update tank displays with real usage data
        const today = new Date().toISOString().split('T')[0];
        const todayUsage = this.sprayingData.dailyUsage[today] || { insecticide: 0, fungicide: 0 };
        
        // Convert ml to liters for display
        const insecticideUsedL = todayUsage.insecticide / 1000;
        const fungicideUsedL = todayUsage.fungicide / 1000;
        
        // Update tank data
        this.appData.tanks.insecticide.usageToday = insecticideUsedL;
        this.appData.tanks.fungicide.usageToday = fungicideUsedL;
        
        // Calculate remaining levels (assuming starting levels)
        const insecticideUsedPercent = (insecticideUsedL / this.appData.tanks.insecticide.capacity) * 100;
        const fungicideUsedPercent = (fungicideUsedL / this.appData.tanks.fungicide.capacity) * 100;
        
        this.appData.tanks.insecticide.currentLevel = Math.max(0, 100 - insecticideUsedPercent);
        this.appData.tanks.fungicide.currentLevel = Math.max(0, 100 - fungicideUsedPercent);
        
        // Update tank status
        this.updateTankStatus();
        
        // Update usage statistics display
        this.displayUsageStatistics();
    }

    displayUsageStatistics() {
        console.log('📊 Usage Statistics:', {
            today: {
                insecticide: `${(this.sprayingData.dailyUsage[new Date().toISOString().split('T')[0]]?.insecticide || 0)}ml`,
                fungicide: `${(this.sprayingData.dailyUsage[new Date().toISOString().split('T')[0]]?.fungicide || 0)}ml`
            },
            total: {
                insecticide: `${this.sprayingData.totalUsage.insecticide}ml`,
                fungicide: `${this.sprayingData.totalUsage.fungicide}ml`
            },
            actionableLogs: this.sprayingData.actionableLogs.length,
            safetyStops: this.sprayingData.unknownLogs.length
        });
    }

    updateSprayingLogs() {
        // You can add UI elements to show recent spraying activity
        console.log('🚿 Recent Actionable Spraying:', this.sprayingData.actionableLogs.slice(0, 5));
        console.log('⚠️ Recent Safety Stops:', this.sprayingData.unknownLogs.slice(0, 5));
    }

    getWeekNumber(date) {
        const onejan = new Date(date.getFullYear(), 0, 1);
        return Math.ceil((((date - onejan) / 86400000) + onejan.getDay() + 1) / 7);
    }

    updateConnectionStatus() {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (statusIndicator && statusText) {
            if (this.isConnectedToMQTT) {
                statusIndicator.className = 'status-indicator connected';
                statusText.textContent = 'Analyzing Data';
                statusIndicator.style.backgroundColor = '#4CAF50';
            } else {
                statusIndicator.className = 'status-indicator disconnected';
                statusText.textContent = 'Disconnected';
                statusIndicator.style.backgroundColor = '#f44336';
            }
        }
    }

    // REMOVE ALL COMMAND FUNCTIONS - ANALYSIS ONLY
    handleEmergency() {
        // Display emergency contact information only
        const emergencyMsg = 'Emergency Contacts:\n• Technical Support: +91-XXXXXXXXXX\n• Agricultural Expert: +91-XXXXXXXXXX';
        this.showNotification('🚨 EMERGENCY - Check system manually!', 'error');
        alert(emergencyMsg);
    }

    // UPDATE EXISTING DISPLAY FUNCTIONS

    updateDiseaseDetection() {
        const diseaseList = document.getElementById('disease-list');
        const diseaseCount = document.getElementById('disease-count');
        
        diseaseCount.textContent = this.appData.diseases.length;
        diseaseList.innerHTML = '';
        
        this.appData.diseases.forEach(disease => {
            const diseaseItem = document.createElement('div');
            diseaseItem.className = 'detection-item';
            
            const timeAgo = this.getTimeAgo(disease.timestamp);
            
            diseaseItem.innerHTML = `
                <div class="detection-header">
                    <div>
                        <div class="detection-title">${disease.name}</div>
                        <div class="detection-id">ID: ${disease.id} • ${timeAgo}</div>
                    </div>
                </div>
                <div class="detection-location">📍 ${disease.location}</div>
            `;
            
            diseaseList.appendChild(diseaseItem);
        });
    }

    updatePestMonitoring() {
        const pestList = document.getElementById('pest-list');
        const pestCount = document.getElementById('pest-count');
        
        pestCount.textContent = this.appData.pests.length;
        pestList.innerHTML = '';
        
        this.appData.pests.forEach(pest => {
            const pestItem = document.createElement('div');
            pestItem.className = 'detection-item';
            
            const timeAgo = this.getTimeAgo(pest.timestamp);
            
            pestItem.innerHTML = `
                <div class="detection-header">
                    <div>
                        <div class="detection-title">${pest.name}</div>
                        <div class="detection-id">ID: ${pest.id} • ${timeAgo}</div>
                    </div>
                </div>
                <div class="detection-location">📍 ${pest.location}</div>
            `;
            
            pestList.appendChild(pestItem);
        });
    }

    getTimeAgo(timestamp) {
        const now = new Date();
        const detected = new Date(timestamp);
        const diffMinutes = Math.floor((now - detected) / (1000 * 60));
        
        if (diffMinutes < 1) return 'Just now';
        if (diffMinutes < 60) return `${diffMinutes} mins ago`;
        
        const diffHours = Math.floor(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours} hours ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays} days ago`;
    }

    handleWeatherUpdate(data) {
        console.log('🌤️ Weather update received:', data);
        if (data.temperature) this.appData.weather.currentTemp = data.temperature;
        if (data.humidity) this.appData.weather.humidity = data.humidity;
        if (data.windSpeed) this.appData.weather.windSpeed = data.windSpeed;
        if (data.condition) this.appData.weather.condition = data.condition;
        if (data.sprayConditions) this.appData.weather.sprayConditions = data.sprayConditions;
        if (data.diseaseRisk) this.appData.weather.diseaseRisk = data.diseaseRisk;
        
        this.updateWeather();
    }

    updateConnectionStatus() {
        // Update the connection status in your existing status indicator
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (statusIndicator && statusText) {
            if (this.isConnectedToMQTT) {
                statusIndicator.className = 'status-indicator connected';
                statusText.textContent = 'Connected';
                statusIndicator.style.backgroundColor = '#4CAF50';
            } else {
                statusIndicator.className = 'status-indicator disconnected';
                statusText.textContent = 'Disconnected';
                statusIndicator.style.backgroundColor = '#f44336';
            }
        }
    }

    // EMERGENCY STOP - SENDS REAL COMMAND TO ARDUINO
    handleEmergency() {
        console.log('🚨 EMERGENCY ACTIVATED');
        this.sendArduinoCommand('none', 'water', 'unknown');
        this.showNotification('🚨 EMERGENCY STOP ACTIVATED! All valves closed.', 'error');
    }

    // WEATHER API FUNCTIONS - ADD THESE RIGHT AFTER handleEmergency()

    async fetchRealWeatherData() {
        try {
            console.log('🌤️ Fetching real weather data...');
            
            const { lat, lon } = this.weatherConfig.location;
            const apiKey = this.weatherConfig.apiKey;
            
            if (!apiKey || apiKey === 'your_openweathermap_api_key_here') {
                console.warn('⚠️ Weather API key not configured');
                return;
            }

            // Fetch current weather
            const currentResponse = await fetch(
                `${this.weatherConfig.endpoints.current}?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`
            );
            
            if (!currentResponse.ok) {
                throw new Error(`Weather API error: ${currentResponse.status}`);
            }
            
            const currentData = await currentResponse.json();
            
            // Fetch 5-day forecast
            const forecastResponse = await fetch(
                `${this.weatherConfig.endpoints.forecast}?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric&cnt=24`
            );
            
            const forecastData = await forecastResponse.json();
            
            // Update your app data with real weather
            this.updateWeatherFromAPI(currentData, forecastData);
            
            console.log('✅ Weather data updated successfully');
            this.showSuccess('Real weather data loaded!');
            
        } catch (error) {
            console.error('❌ Weather API Error:', error);
            this.showError('Failed to update weather data');
        }
    }

    updateWeatherFromAPI(currentData, forecastData) {
        // Update current weather
        this.appData.weather = {
            ...this.appData.weather, // Keep existing properties
            location: `${currentData.name}, Tamil Nadu`,
            currentTemp: Math.round(currentData.main.temp),
            condition: this.mapWeatherCondition(currentData.weather[0].main),
            humidity: currentData.main.humidity,
            windSpeed: Math.round(currentData.wind.speed * 3.6), // Convert m/s to km/h
            pressure: currentData.main.pressure,
            visibility: Math.round((currentData.visibility || 10000) / 1000), // Convert to km
            
            // Calculate spray conditions based on real data
            sprayConditions: this.calculateSprayConditions(
                currentData.wind.speed * 3.6, 
                currentData.main.humidity, 
                currentData.main.temp
            ),
            
            // Calculate disease risk
            diseaseRisk: this.calculateDiseaseRisk(
                currentData.main.humidity,
                currentData.main.temp,
                currentData.weather[0].main
            ),
            
            // Process forecast
            forecast: this.processForecastData(forecastData.list),
            
            // Generate weather alerts
            alerts: this.generateWeatherAlerts(currentData, forecastData.list)
        };
        
        // Update the dashboard
        this.updateWeather();
        
        // Send weather update via MQTT if connected
        if (this.isConnectedToMQTT) {
            this.sendWeatherUpdateToMQTT();
        }
    }

    mapWeatherCondition(openWeatherCondition) {
        const conditionMap = {
            'Clear': 'sunny',
            'Clouds': 'cloudy',
            'Rain': 'light_rain',
            'Drizzle': 'light_rain',
            'Thunderstorm': 'thunderstorm',
            'Snow': 'snow',
            'Mist': 'partly_cloudy',
            'Smoke': 'partly_cloudy',
            'Haze': 'partly_cloudy',
            'Dust': 'partly_cloudy',
            'Fog': 'cloudy',
            'Sand': 'partly_cloudy',
            'Ash': 'cloudy',
            'Squall': 'thunderstorm',
            'Tornado': 'thunderstorm'
        };
        
        return conditionMap[openWeatherCondition] || 'partly_cloudy';
    }

    calculateSprayConditions(windSpeed, humidity, temperature) {
        // Agricultural spray condition logic
        if (windSpeed > 15) return 'unfavorable'; // Too windy
        if (windSpeed < 3) return 'poor'; // Too calm (drift risk)
        if (humidity > 85) return 'unfavorable'; // Too humid
        if (humidity > 75) return 'poor'; // High humidity
        if (temperature < 10 || temperature > 35) return 'unfavorable'; // Temperature extremes
        
        return 'favorable';
    }

    calculateDiseaseRisk(humidity, temperature, condition) {
        let riskScore = 0;
        
        // High humidity increases disease risk
        if (humidity > 80) riskScore += 2;
        else if (humidity > 70) riskScore += 1;
        
        // Temperature range favorable for diseases (20-30°C)
        if (temperature >= 20 && temperature <= 30) riskScore += 1;
        
        // Rain increases disease risk
        if (condition.includes('Rain') || condition.includes('Drizzle')) riskScore += 2;
        
        if (riskScore >= 4) return 'high';
        if (riskScore >= 2) return 'medium';
        return 'low';
    }

    processForecastData(forecastList) {
        const dailyForecasts = {};
        
        // Group forecast by date
        forecastList.slice(0, 24).forEach(item => { // Next 3 days (8 forecasts per day)
            const date = new Date(item.dt * 1000).toISOString().split('T')[0];
            
            if (!dailyForecasts[date]) {
                dailyForecasts[date] = {
                    date: date,
                    high: item.main.temp,
                    low: item.main.temp,
                    conditions: [],
                    humidity: item.main.humidity,
                    windSpeed: item.wind.speed * 3.6,
                    precipitation: item.rain ? item.rain['3h'] || 0 : 0
                };
            }
            
            const day = dailyForecasts[date];
            day.high = Math.max(day.high, item.main.temp);
            day.low = Math.min(day.low, item.main.temp);
            day.conditions.push(item.weather[0].main);
        });
        
        // Convert to array and format
        return Object.values(dailyForecasts).slice(0, 3).map(day => ({
            date: day.date,
            high: Math.round(day.high),
            low: Math.round(day.low),
            condition: this.mapWeatherCondition(this.getMostCommonCondition(day.conditions)),
            humidity: Math.round(day.humidity),
            windSpeed: Math.round(day.windSpeed),
            precipitation: day.precipitation
        }));
    }

    getMostCommonCondition(conditions) {
        const counts = {};
        conditions.forEach(condition => {
            counts[condition] = (counts[condition] || 0) + 1;
        });
        
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }

    generateWeatherAlerts(currentData, forecastList) {
        const alerts = [];
        const windSpeed = currentData.wind.speed * 3.6;
        const humidity = currentData.main.humidity;
        const temp = currentData.main.temp;
        
        // Wind speed alert
        if (windSpeed > 15) {
            alerts.push({
                type: 'wind_warning',
                severity: 'high',
                message: `High wind speed (${Math.round(windSpeed)} km/h). Avoid spraying operations.`,
                validUntil: new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString()
            });
        }
        
        // Rain alert
        const rainyPeriods = forecastList.filter(item => 
            item.weather[0].main.includes('Rain') || item.weather[0].main.includes('Drizzle')
        );
        
        if (rainyPeriods.length > 0) {
            const nextRain = new Date(rainyPeriods[0].dt * 1000);
            alerts.push({
                type: 'rain_forecast',
                severity: 'medium',
                message: `Rain expected within 24 hours. Plan treatments accordingly.`,
                validUntil: nextRain.toISOString()
            });
        }
        
        // Temperature extreme alert
        if (temp > 35) {
            alerts.push({
                type: 'temperature_extreme',
                severity: 'medium',
                message: `High temperature (${Math.round(temp)}°C). Monitor crop stress and avoid midday spraying.`,
                validUntil: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString()
            });
        } else if (temp < 5) {
            alerts.push({
                type: 'temperature_extreme',
                severity: 'high',
                message: `Low temperature (${Math.round(temp)}°C). Protect crops from frost damage.`,
                validUntil: new Date(Date.now() + 12 * 60 * 60 * 1000).toISOString()
            });
        }
        
        return alerts;
    }

    sendWeatherUpdateToMQTT() {
        if (this.mqttClient && this.isConnectedToMQTT) {
            const weatherUpdate = {
                temperature: this.appData.weather.currentTemp,
                humidity: this.appData.weather.humidity,
                windSpeed: this.appData.weather.windSpeed,
                condition: this.appData.weather.condition,
                sprayConditions: this.appData.weather.sprayConditions,
                diseaseRisk: this.appData.weather.diseaseRisk,
                timestamp: new Date().toISOString()
            };
            
            this.mqttClient.publish(this.mqttConfig.topics.weather, JSON.stringify(weatherUpdate), { qos: 1 });
            console.log('📤 Weather update sent to MQTT');
        }
    }

    startWeatherUpdates() {
        // Initial weather fetch
        this.fetchRealWeatherData();
        
        // Set up periodic updates
        this.weatherUpdateTimer = setInterval(() => {
            this.fetchRealWeatherData();
        }, this.weatherConfig.updateInterval);
        
        console.log('🌤️ Weather updates started (every 5 minutes)');
    }

    stopWeatherUpdates() {
        if (this.weatherUpdateTimer) {
            clearInterval(this.weatherUpdateTimer);
            this.weatherUpdateTimer = null;
            console.log('🌤️ Weather updates stopped');
        }
    }

    // PWA Service Worker Registration
    registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            const swCode = `
                const CACHE_NAME = 'smart-farm-v2';
                const urlsToCache = [
                    '/',
                    '/style.css',
                    '/app.js',
                    'https://fonts.googleapis.com/icon?family=Material+Icons'
                ];

                self.addEventListener('install', event => {
                    event.waitUntil(
                        caches.open(CACHE_NAME)
                            .then(cache => cache.addAll(urlsToCache))
                    );
                });

                self.addEventListener('fetch', event => {
                    event.respondWith(
                        caches.match(event.request)
                            .then(response => {
                                if (response) {
                                    return response;
                                }
                                return fetch(event.request);
                            })
                    );
                });
            `;

            const blob = new Blob([swCode], { type: 'application/javascript' });
            const swUrl = URL.createObjectURL(blob);
            
            navigator.serviceWorker.register(swUrl)
                .then(registration => {
                    console.log('SW registered: ', registration);
                })
                .catch(registrationError => {
                    console.log('SW registration failed: ', registrationError);
                });
        }
    }

    // Setup Event Listeners
    setupEventListeners() {
        // Authentication
        document.getElementById('send-otp-btn').addEventListener('click', () => this.sendOTP());
        document.getElementById('verify-otp-btn').addEventListener('click', () => this.verifyOTP());
        document.getElementById('resend-otp-btn').addEventListener('click', () => this.resendOTP());

        // OTP Input handling
        const otpInputs = document.querySelectorAll('.otp-digit');
        otpInputs.forEach((input, index) => {
            input.addEventListener('input', (e) => this.handleOTPInput(e, index));
            input.addEventListener('keydown', (e) => this.handleOTPKeydown(e, index));
        });

        // Language toggle
        document.getElementById('language-toggle').addEventListener('click', () => this.toggleLanguage());

        // Emergency button - NOW SENDS REAL MQTT COMMAND
        document.getElementById('emergency-btn').addEventListener('click', () => this.handleEmergency());

        // Weather alert dismiss
        document.getElementById('dismiss-alert').addEventListener('click', () => this.dismissWeatherAlert());

        // User menu
        document.querySelector('.user-avatar').addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleUserMenu();
        });
        document.getElementById('logout-btn').addEventListener('click', () => this.logout());

        // Tank refill
        document.getElementById('refill-all-btn').addEventListener('click', () => this.refillAllTanks());

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            const userMenu = document.querySelector('.user-menu');
            const dropdown = document.querySelector('.user-dropdown');
            if (!userMenu.contains(e.target)) {
                dropdown.classList.add('hidden');
            }
        });

        // Pull to refresh simulation
        this.setupPullToRefresh();

        // NEW: Add valve control buttons for testing
        this.addValveControlButtons();
    }

    // NEW: Add valve control buttons for testing
    addValveControlButtons() {
        // Create test buttons after authentication
        setTimeout(() => {
            if (this.isAuthenticated) {
                const emergencyBtn = document.getElementById('emergency-btn');
                if (emergencyBtn && !document.getElementById('valve-test-controls')) {
                    const testControls = document.createElement('div');
                    testControls.id = 'valve-test-controls';
                    testControls.style.cssText = `
                        position: fixed;
                        bottom: 20px;
                        left: 20px;
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        display: flex;
                        gap: 10px;
                        flex-wrap: wrap;
                        z-index: 1000;
                        max-width: 300px;
                    `;
                    
                    testControls.innerHTML = `
                        <div style="width: 100%; font-weight: bold; margin-bottom: 8px; color: #333;">Arduino Control</div>
                        <button id="test-water-a" style="padding: 8px 12px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Water A</button>
                        <button id="test-water-b" style="padding: 8px 12px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Water B</button>
                        <button id="test-chem-a" style="padding: 8px 12px; background: #FF9800; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Chem A</button>
                        <button id="test-chem-b" style="padding: 8px 12px; background: #FF9800; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Chem B</button>
                        <button id="test-close-all" style="padding: 8px 12px; background: #757575; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Close All</button>
                    `;
                    
                    document.body.appendChild(testControls);
                    
                    // Add event listeners
                    document.getElementById('test-water-a').addEventListener('click', () => this.sendArduinoCommand('A', 'water', 'actionable'));
                    document.getElementById('test-water-b').addEventListener('click', () => this.sendArduinoCommand('B', 'water', 'actionable'));
                    document.getElementById('test-chem-a').addEventListener('click', () => this.sendArduinoCommand('A', 'chem', 'actionable'));
                    document.getElementById('test-chem-b').addEventListener('click', () => this.sendArduinoCommand('B', 'chem', 'actionable'));
                    document.getElementById('test-close-all').addEventListener('click', () => this.sendArduinoCommand('none', 'water', 'unknown'));
                }
            }
        }, 5000);
    }

    // Splash Screen
    showSplashScreen() {
        setTimeout(() => {
            document.getElementById('splash-screen').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('splash-screen').classList.add('hidden');
                this.showAuthScreen();
            }, 300);
        }, 2500);
    }

    // Authentication Flow
    showAuthScreen() {
        document.getElementById('auth-screen').classList.remove('hidden');
        document.getElementById('auth-screen').classList.add('fade-in');
    }

    sendOTP() {
        const phoneInput = document.getElementById('phone-input');
        const phone = phoneInput.value.trim();
        
        if (!this.validatePhoneNumber(phone)) {
            this.showError('Please enter a valid 10-digit phone number');
            return;
        }

        // Simulate OTP sending
        document.getElementById('phone-step').classList.add('hidden');
        document.getElementById('otp-step').classList.remove('hidden');
        document.getElementById('phone-display').textContent = `+91 ${phone}`;
        
        // Auto-fill OTP for demo (123456)
        setTimeout(() => {
            const otpInputs = document.querySelectorAll('.otp-digit');
            const demoOTP = '123456';
            otpInputs.forEach((input, index) => {
                input.value = demoOTP[index];
            });
        }, 1000);
        
        this.showSuccess('OTP sent successfully!');
    }

    verifyOTP() {
        const otpInputs = document.querySelectorAll('.otp-digit');
        const otp = Array.from(otpInputs).map(input => input.value).join('');
        
        if (otp.length !== 6) {
            this.showError('Please enter complete OTP');
            return;
        }

        if (otp === '123456') {
            this.isAuthenticated = true;
            this.userData = {
                phone: document.getElementById('phone-display').textContent
            };
            this.showMainApp();
        } else {
            this.showError('Invalid OTP. Please try again.');
        }
    }

    resendOTP() {
        this.showSuccess('OTP resent successfully!');
    }

    validatePhoneNumber(phone) {
        const cleanPhone = phone.replace(/\D/g, '');
        return cleanPhone.length === 10 && /^[6-9]\d{9}$/.test(cleanPhone);
    }

    handleOTPInput(e, index) {
        const input = e.target;
        const value = input.value;
        
        if (value.length === 1 && index < 5) {
            document.querySelectorAll('.otp-digit')[index + 1].focus();
        }
    }

    handleOTPKeydown(e, index) {
        if (e.key === 'Backspace' && e.target.value === '' && index > 0) {
            document.querySelectorAll('.otp-digit')[index - 1].focus();
        }
    }

    // Main Application
    showMainApp() {
        document.getElementById('auth-screen').classList.add('hidden');
        document.getElementById('app-container').classList.remove('hidden');
        document.getElementById('app-container').classList.add('fade-in');
        
        this.updateUserDisplay();
        this.updateDashboard();
        this.translateUI();
        this.checkWeatherAlerts();
        this.addValveControlButtons(); // Add test buttons

        this.connectToMQTT();
        
        // START WEATHER UPDATES (if you added weather API)
        if (this.startWeatherUpdates) {
            this.startWeatherUpdates();
        }

    }

    updateUserDisplay() {
        if (this.userData) {
            document.getElementById('user-phone-display').textContent = this.userData.phone;
        }
    }

    // Language Management
    toggleLanguage() {
        this.currentLanguage = this.currentLanguage === 'en' ? 'ta' : 'en';
        this.updateLanguageToggle();
        this.translateUI();
        this.updateDashboard(); // Refresh to show translated content
    }

    updateLanguageToggle() {
        const toggle = document.getElementById('language-toggle');
        const flagIcon = toggle.querySelector('.flag-icon');
        const languageText = toggle.querySelector('.language-text');
        
        if (this.currentLanguage === 'en') {
            flagIcon.textContent = '🇬🇧';
            languageText.textContent = 'EN';
        } else {
            flagIcon.textContent = '🇮🇳';
            languageText.textContent = 'தமிழ்';
        }
    }

    translateUI() {
        const elements = document.querySelectorAll('[data-translate]');
        elements.forEach(element => {
            const key = element.getAttribute('data-translate');
            if (this.translations[this.currentLanguage][key]) {
                element.textContent = this.translations[this.currentLanguage][key];
            }
        });
        
        document.title = this.translations[this.currentLanguage].appTitle;
    }

    // Dashboard Updates
    updateDashboard() {
        this.updateFarmHealth();
        this.updateTankStatus();
        this.updateDiseaseDetection();
        this.updatePestMonitoring();
        this.updateWeather();
        this.updateRecommendations();
    }

    updateFarmHealth() {
        const data = this.appData.farmHealth;
        
        document.getElementById('active-diseases').textContent = data.activeDiseases;
        document.getElementById('active-pests').textContent = data.activePests;
        document.getElementById('treatment-effectiveness').textContent = `${data.treatmentEffectiveness}%`;
        
        // Update status indicator
        const statusIndicator = document.getElementById('overall-status-indicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        
        statusDot.className = 'status-dot';
        if (data.overallStatus === 'good') {
            statusDot.classList.add('success');
            statusText.textContent = this.translations[this.currentLanguage].good || 'Good';
        } else if (data.overallStatus === 'attention') {
            statusDot.classList.add('warning');
            statusText.textContent = this.translations[this.currentLanguage].attention || 'Attention Required';
        } else {
            statusDot.classList.add('error');
            statusText.textContent = this.translations[this.currentLanguage].critical || 'Critical';
        }
    }

    updateTankStatus() {
        const tanks = this.appData.tanks;
        
        // Update insecticide tank
        this.updateTank('insecticide', tanks.insecticide);
        this.updateTank('fungicide', tanks.fungicide);
        this.updateTank('water', tanks.water);
    }

    // Utility function to format numbers properly
    formatNumber(num, decimals = 1) {
        return Number(num).toFixed(decimals).replace(/\.0+$/, '');
    }

    formatInteger(num) {
        return Math.round(Number(num));
    }

    updateTank(tankType, tankData) {
        const level = document.getElementById(`${tankType}-level`);
        const percentage = document.getElementById(`${tankType}-percentage`);
        const liters = document.getElementById(`${tankType}-liters`);
        const status = document.getElementById(`${tankType}-status`);
        
        // Format all numeric values properly
        const levelPercent = this.formatInteger(tankData.currentLevel);
        const currentLiters = this.formatNumber(tankData.currentLiters, 1);
        const capacity = this.formatInteger(tankData.capacity);
        
        level.style.height = `${levelPercent}%`;
        percentage.textContent = `${levelPercent}%`;
        liters.textContent = `${currentLiters}L / ${capacity}L`;
        
        // Update status
        status.className = `tank-status ${tankData.status}`;
        status.textContent = tankData.status.charAt(0).toUpperCase() + tankData.status.slice(1);
        
        // Update tank-specific details
        if (tankType === 'insecticide' || tankType === 'fungicide') {
            const ingredient = document.getElementById(`${tankType}-ingredient`);
            const usage = document.getElementById(`${tankType}-usage`);
            ingredient.textContent = tankData.activeIngredient;
            usage.textContent = `${this.formatNumber(tankData.usageToday, 1)}L`;
        } else if (tankType === 'water') {
            const quality = document.getElementById('water-quality');
            const flow = document.getElementById('water-flow');
            quality.textContent = tankData.quality.charAt(0).toUpperCase() + tankData.quality.slice(1);
            flow.textContent = `${this.formatNumber(tankData.flowRate, 1)} L/min`;
        }
    }

    updateDiseaseDetection() {
        const diseaseList = document.getElementById('disease-list');
        const diseaseCount = document.getElementById('disease-count');
        
        diseaseCount.textContent = this.appData.diseases.length;
        diseaseList.innerHTML = '';
        
        this.appData.diseases.forEach(disease => {
            const diseaseItem = document.createElement('div');
            diseaseItem.className = `detection-item severity-${disease.severity}`;
            
            const displayName = this.currentLanguage === 'ta' ? disease.nameTamil : disease.name;
            
            diseaseItem.innerHTML = `
                <div class="detection-header">
                    <div>
                        <div class="detection-title">${displayName}</div>
                        <div class="detection-id">ID: ${disease.id}</div>
                    </div>
                    <div class="detection-severity ${disease.severity}">${disease.severity.toUpperCase()}</div>
                </div>
                <div class="detection-location">${disease.location}</div>
                <div class="detection-recommendation">${disease.recommendation}</div>
            `;
            
            diseaseList.appendChild(diseaseItem);
        });
    }

    updatePestMonitoring() {
        const pestList = document.getElementById('pest-list');
        const pestCount = document.getElementById('pest-count');
        
        pestCount.textContent = this.appData.pests.length;
        pestList.innerHTML = '';
        
        this.appData.pests.forEach(pest => {
            const pestItem = document.createElement('div');
            pestItem.className = `detection-item severity-${pest.severity}`;
            
            const displayName = this.currentLanguage === 'ta' ? pest.nameTamil : pest.name;
            
            pestItem.innerHTML = `
                <div class="detection-header">
                    <div>
                        <div class="detection-title">${displayName}</div>
                        <div class="detection-id">ID: ${pest.id}</div>
                    </div>
                    <div class="detection-severity ${pest.severity}">${pest.severity.toUpperCase()}</div>
                </div>
                <div class="detection-location">${pest.location} - ${pest.population}</div>
                <div class="detection-recommendation">${pest.recommendation}</div>
            `;
            
            pestList.appendChild(pestItem);
        });
    }

    updateWeather() {
        const weather = this.appData.weather;
        
        document.getElementById('weather-location').textContent = weather.location;
        document.getElementById('current-temp').textContent = `${weather.currentTemp}°C`;
        document.getElementById('humidity').textContent = `${weather.humidity}%`;
        document.getElementById('wind-speed').textContent = `${weather.windSpeed} km/h`;
        
        // Update weather icon
        const weatherIcon = document.querySelector('.weather-icon');
        const conditionText = document.querySelector('.weather-condition span');
        
        switch (weather.condition) {
            case 'partly_cloudy':
                weatherIcon.textContent = 'partly_sunny';
                conditionText.textContent = 'Partly Cloudy';
                break;
            case 'sunny':
                weatherIcon.textContent = 'wb_sunny';
                conditionText.textContent = 'Sunny';
                break;
            case 'cloudy':
                weatherIcon.textContent = 'cloud';
                conditionText.textContent = 'Cloudy';
                break;
            case 'light_rain':
                weatherIcon.textContent = 'grain';
                conditionText.textContent = 'Light Rain';
                break;
        }
        
        // Update indicators
        const sprayConditions = document.getElementById('spray-conditions');
        const diseaseRisk = document.getElementById('disease-risk');
        
        sprayConditions.className = `indicator-status ${weather.sprayConditions}`;
        sprayConditions.textContent = weather.sprayConditions.charAt(0).toUpperCase() + weather.sprayConditions.slice(1);
        
        diseaseRisk.className = `indicator-status ${weather.diseaseRisk}`;
        diseaseRisk.textContent = weather.diseaseRisk.charAt(0).toUpperCase() + weather.diseaseRisk.slice(1);
        
        // Update forecast
        this.updateForecast(weather.forecast);
    }

    updateForecast(forecast) {
        const forecastContainer = document.getElementById('weather-forecast');
        forecastContainer.innerHTML = '';
        
        forecast.slice(0, 3).forEach(day => {
            const forecastItem = document.createElement('div');
            forecastItem.className = 'forecast-item';
            
            const date = new Date(day.date);
            const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
            
            let icon;
            switch (day.condition) {
                case 'sunny': icon = 'wb_sunny'; break;
                case 'cloudy': icon = 'cloud'; break;
                case 'light_rain': icon = 'grain'; break;
                default: icon = 'wb_sunny';
            }
            
            forecastItem.innerHTML = `
                <div class="forecast-date">${dayName}</div>
                <i class="material-icons forecast-icon">${icon}</i>
                <div class="forecast-temps">${day.high}°/${day.low}°</div>
            `;
            
            forecastContainer.appendChild(forecastItem);
        });
    }

    updateRecommendations() {
        const recommendationsList = document.getElementById('recommendations-list');
        const recommendationsCount = document.getElementById('recommendations-count');
        
        recommendationsCount.textContent = this.appData.recommendations.length;
        recommendationsList.innerHTML = '';
        
        this.appData.recommendations.forEach(rec => {
            const recItem = document.createElement('div');
            recItem.className = `recommendation-item priority-${rec.priority}`;
            
            recItem.innerHTML = `
                <div class="recommendation-header">
                    <div>
                        <div class="recommendation-title">${rec.title}</div>
                    </div>
                    <div class="recommendation-type">${rec.type.replace('_', ' ').toUpperCase()}</div>
                </div>
                <div class="recommendation-description">${rec.description}</div>
                <div class="recommendation-details">
                    <div class="recommendation-detail">
                        <strong>Treatment:</strong>
                        <span>${rec.treatment}</span>
                    </div>
                    <div class="recommendation-detail">
                        <strong>Dosage:</strong>
                        <span>${rec.dosage}</span>
                    </div>
                    <div class="recommendation-detail">
                        <strong>Timing:</strong>
                        <span>${rec.timing}</span>
                    </div>
                    <div class="recommendation-detail">
                        <strong>Safety Period:</strong>
                        <span>${rec.safetyPeriod}</span>
                    </div>
                </div>
            `;
            
            recommendationsList.appendChild(recItem);
        });
    }

    // Weather Alerts
    checkWeatherAlerts() {
        const alerts = this.appData.weather.alerts;
        if (alerts && alerts.length > 0) {
            const banner = document.getElementById('weather-alert-banner');
            const message = document.getElementById('weather-alert-message');
            
            message.textContent = alerts[0].message;
            banner.classList.remove('hidden');
        }
    }

    dismissWeatherAlert() {
        document.getElementById('weather-alert-banner').classList.add('hidden');
    }

    // Tank Management
    refillAllTanks() {
        // Simulate tank refilling with proper number formatting
        this.appData.tanks.insecticide.currentLevel = 100;
        this.appData.tanks.insecticide.currentLiters = this.appData.tanks.insecticide.capacity;
        this.appData.tanks.insecticide.status = 'normal';
        
        this.appData.tanks.fungicide.currentLevel = 100;
        this.appData.tanks.fungicide.currentLiters = this.appData.tanks.fungicide.capacity;
        this.appData.tanks.fungicide.status = 'normal';
        
        this.appData.tanks.water.currentLevel = 100;
        this.appData.tanks.water.currentLiters = this.appData.tanks.water.capacity;
        this.appData.tanks.water.status = 'normal';
        
        this.updateTankStatus();
        this.showSuccess('All tanks refilled successfully!');
    }

    // Data Simulation
    startDataSimulation() {
        setInterval(() => {
            if (this.isAuthenticated) {
                this.simulateDataChanges();
                this.updateTankStatus();
            }
        }, 60000); // Update every minute
    }

    simulateDataChanges() {
        // Simulate tank usage with proper number handling
        const tanks = this.appData.tanks;
        
        // Insecticide usage
        if (tanks.insecticide.currentLevel > 5) {
            const usage = Math.random() * 0.5;
            tanks.insecticide.currentLevel = Math.max(5, tanks.insecticide.currentLevel - usage);
            tanks.insecticide.currentLiters = Math.round((tanks.insecticide.currentLevel / 100) * tanks.insecticide.capacity * 10) / 10;
            tanks.insecticide.usageToday = Math.round((tanks.insecticide.usageToday + usage * 2) * 10) / 10;
            
            if (tanks.insecticide.currentLevel < 20) {
                tanks.insecticide.status = 'critical';
            } else if (tanks.insecticide.currentLevel < 40) {
                tanks.insecticide.status = 'attention';
            }
        }
        
        // Similar for fungicide
        if (tanks.fungicide.currentLevel > 5) {
            const usage = Math.random() * 0.3;
            tanks.fungicide.currentLevel = Math.max(5, tanks.fungicide.currentLevel - usage);
            tanks.fungicide.currentLiters = Math.round((tanks.fungicide.currentLevel / 100) * tanks.fungicide.capacity * 10) / 10;
            tanks.fungicide.usageToday = Math.round((tanks.fungicide.usageToday + usage * 1.5) * 10) / 10;
            
            if (tanks.fungicide.currentLevel < 20) {
                tanks.fungicide.status = 'critical';
            } else if (tanks.fungicide.currentLevel < 40) {
                tanks.fungicide.status = 'attention';
            }
        }
        
        // Water usage
        if (tanks.water.currentLevel > 10) {
            const usage = Math.random() * 0.8;
            tanks.water.currentLevel = Math.max(10, tanks.water.currentLevel - usage);
            tanks.water.currentLiters = Math.round((tanks.water.currentLevel / 100) * tanks.water.capacity * 10) / 10;
            
            if (tanks.water.currentLevel < 30) {
                tanks.water.status = 'attention';
            }
        }
    }

    // User Interface Helpers
    toggleUserMenu() {
        const dropdown = document.querySelector('.user-dropdown');
        dropdown.classList.toggle('hidden');
    }

    logout() {
        this.isAuthenticated = false;
        this.userData = null;
        
        // Disconnect MQTT
        if (this.mqttClient) {
            this.mqttClient.end();
            this.isConnectedToMQTT = false;
        }
        
        // Remove test controls
        const testControls = document.getElementById('valve-test-controls');
        if (testControls) {
            testControls.remove();
        }
        
        document.querySelector('.user-dropdown').classList.add('hidden');
        document.getElementById('app-container').classList.add('hidden');
        document.getElementById('auth-screen').classList.remove('hidden');
        
        // Reset auth form
        document.getElementById('otp-step').classList.add('hidden');
        document.getElementById('phone-step').classList.remove('hidden');
        document.getElementById('phone-input').value = '';
        document.querySelectorAll('.otp-digit').forEach(input => input.value = '');
        this.stopWeatherUpdates();
    }

    setupPullToRefresh() {
        let startY = 0;
        let currentY = 0;
        let pullDistance = 0;
        const threshold = 80;
        
        document.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchmove', (e) => {
            if (window.scrollY === 0) {
                currentY = e.touches[0].clientY;
                pullDistance = currentY - startY;
                
                if (pullDistance > 0) {
                    e.preventDefault();
                }
            }
        });
        
        document.addEventListener('touchend', () => {
            if (pullDistance > threshold) {
                this.refreshData();
            }
            pullDistance = 0;
        });
    }

    refreshData() {
        this.showSuccess('Data refreshed!');
        this.simulateDataChanges();
        this.updateDashboard();
    }

    // Utility Functions
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification--${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#4CAF50' : type === 'warning' ? '#FF9800' : '#f44336'};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
            word-wrap: break-word;
            font-weight: bold;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SmartFarmApp();
});

// Add notification animations to document
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
