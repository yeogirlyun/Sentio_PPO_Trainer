# Sentio Trader Project Rules

**Established:** 2025-01-27  
**Status:** Mandatory for all AI models and contributors

---

## 📋 Documentation Policy

### **CRITICAL RULE: Two-Document System**

Sentio Trader maintains **EXACTLY TWO** documentation files:

1. **`docs/ARCHITECTURE.md`** - Complete system architecture and technical details
2. **`docs/README.md`** - Installation, usage, and user guide

### **Mandatory Documentation Rules**

#### ✅ **ALLOWED Actions**
- **UPDATE** existing content in `docs/ARCHITECTURE.md`
- **UPDATE** existing content in `docs/README.md`
- **REPLACE** outdated information with current codebase reflection
- **ENHANCE** existing sections with new features

#### ❌ **FORBIDDEN Actions**
- **CREATE** new `.md` files anywhere in the project
- **CREATE** additional README files in subdirectories
- **CREATE** separate architecture documents
- **CREATE** feature-specific documentation files
- **LEAVE** outdated information in documentation

### **Documentation Update Requirements**

When making code changes, AI models **MUST**:

1. **Update Architecture**: Reflect changes in `docs/ARCHITECTURE.md`
2. **Update Usage**: Reflect changes in `docs/README.md`
3. **Remove Outdated**: Delete obsolete information
4. **Keep Current**: Ensure documentation matches current codebase
5. **No New Files**: Never create additional documentation files

---

## 🏗️ Architecture Rules

### **System Architecture Principles**

1. **Multi-Algorithm Engine**: All trading logic goes through the multi-algorithm system
2. **Kafka Messaging**: All inter-service communication uses Kafka
3. **Event-Driven**: Asynchronous, non-blocking operations
4. **GUI Integration**: All features must have GUI controls
5. **Performance Tracking**: All algorithms must have performance monitoring

### **Code Organization**

```
Sentio/
├── docs/                    # ONLY these two files allowed
│   ├── ARCHITECTURE.md      # Technical architecture
│   └── README.md           # User guide and installation
├── services/               # Core trading services
├── ui/                     # GUI components
├── models/                 # AI/ML models
├── core/                   # System controllers
├── data/                   # Data management
├── training/               # Algorithm training
├── backtesting/           # Performance testing
└── analysis/              # Results storage (no docs)
```

### **Component Integration Rules**

1. **New Algorithms**: Must integrate with `multi_algorithm_signal_processor.py`
2. **GUI Features**: Must integrate with existing tab system
3. **Performance Tracking**: Must provide real-time metrics
4. **Configuration**: Must use existing config system
5. **Logging**: Must use centralized logging system (see Logging Policy below)

---

## 🤖 AI Model Guidelines

### **When Working on Sentio Trader**

#### **Documentation Updates (MANDATORY)**
```python
# After any code changes, ALWAYS update:
1. docs/ARCHITECTURE.md - Technical changes
2. docs/README.md - User-facing changes

# NEVER create:
- New .md files
- Additional README files
- Feature-specific documentation
- Temporary documentation files
```

#### **Code Changes**
```python
# Follow these patterns:
1. Integrate with existing multi-algorithm system
2. Add GUI controls for new features
3. Implement performance tracking
4. Use existing Kafka messaging
5. Follow established error handling
```

#### **Testing Requirements**
```python
# Before completing work:
1. Test GUI integration
2. Verify algorithm performance tracking
3. Confirm Kafka message flow
4. Validate configuration system
5. Update both documentation files
```

---

## 🚫 Code Duplication Prevention Rules

### **CRITICAL PRINCIPLE: No Duplicate Modules**

**Code duplication is pure evil and must be avoided at all costs.**

#### **File Naming Rules (MANDATORY)**

##### ✅ **ALLOWED Naming Patterns**
```python
# Descriptive, specific names that indicate exact purpose:
ppo_trainer.py          # PPO training system
market_data_producer.py # Market data production
risk_manager.py         # Risk management
signal_processor.py     # Signal processing
chart_widget.py         # Chart display widget
```

##### ❌ **FORBIDDEN Naming Patterns**
```python
# Vague adjectives that create confusion:
advanced_*.py          # What makes it "advanced"?
enhanced_*.py          # Enhanced compared to what?
optimized_*.py         # All code should be optimized
improved_*.py          # Improved from what version?
fixed_*.py             # Fixes should overwrite, not duplicate
v2_*.py, v3_*.py       # Version numbers in filenames
final_*.py             # Nothing is ever truly final
new_*.py               # Everything was new once
better_*.py            # Subjective and meaningless
```

#### **Module Evolution Rules**

##### **Rule 1: Overwrite, Don't Duplicate**
```python
# WRONG: Creating new versions
ppo_trainer.py          # Original
advanced_ppo_trainer.py # ❌ FORBIDDEN - creates confusion
enhanced_ppo_trainer.py # ❌ FORBIDDEN - which one to use?

# RIGHT: Evolve in place
ppo_trainer.py          # ✅ Single source of truth
# When improving: edit ppo_trainer.py directly
```

##### **Rule 2: Specific Names for Different Behavior**
```python
# WRONG: Vague adjectives
signal_processor.py         # Original
advanced_signal_processor.py # ❌ What's "advanced"?

# RIGHT: Specific characteristics
signal_processor.py         # General signal processing
momentum_signal_processor.py # ✅ Momentum-based signals
ml_signal_processor.py      # ✅ Machine learning signals
```

##### **Rule 3: Temporary Files Must Be Cleaned**
```python
# During development, temporary files are acceptable:
debug_*.py              # For debugging only
test_*.py               # For testing only
temp_*.py               # For temporary work

# But MUST be removed before completion:
rm debug_*.py           # Clean up when done
rm test_*.py            # Remove temporary tests
rm temp_*.py            # Delete temporary files
```

#### **Implementation Guidelines**

##### **When Improving Existing Code:**
1. **Edit the original file directly**
2. **Do NOT create new versions with adjectives**
3. **Use git for version history, not filenames**
4. **Test thoroughly before overwriting**
5. **Update imports if class names change**

##### **When Adding New Functionality:**
1. **Ask: Is this truly different behavior?**
2. **If same purpose: enhance existing file**
3. **If different purpose: use specific descriptive name**
4. **Never use vague adjectives like "advanced" or "enhanced"**

##### **Examples of Correct Evolution:**

```python
# Scenario: Improving PPO trainer
# WRONG:
ppo_trainer.py              # Original
advanced_ppo_trainer.py     # ❌ Creates confusion

# RIGHT:
ppo_trainer.py              # ✅ Evolved in place

# Scenario: Adding different signal processing
# WRONG:
signal_processor.py         # Original  
enhanced_signal_processor.py # ❌ Vague adjective

# RIGHT:
signal_processor.py         # Base processor
momentum_signal_processor.py # ✅ Specific: momentum-based
mean_reversion_processor.py  # ✅ Specific: mean reversion
```

#### **Enforcement Rules**

##### **AI Models MUST:**
1. **Check for existing similar files before creating new ones**
2. **Use specific, descriptive names that indicate exact purpose**
3. **Never use vague adjectives (advanced, enhanced, optimized, etc.)**
4. **Overwrite existing files when improving functionality**
5. **Remove temporary/debug files after completion**
6. **Update all imports when renaming files**

##### **Automatic Violations:**
Any file with these patterns will be **automatically rejected**:
- `*advanced*`
- `*enhanced*`
- `*optimized*`
- `*improved*`
- `*fixed*`
- `*v2*`, `*v3*`, etc.
- `*final*`
- `*new*`
- `*better*`

#### **Code Review Checklist**

Before completing any work, verify:
- [ ] No duplicate modules with similar functionality
- [ ] No vague adjectives in filenames
- [ ] All temporary/debug files removed
- [ ] Imports updated for any renamed files
- [ ] Single source of truth for each functionality
- [ ] File names clearly indicate specific purpose
- [ ] **Run duplicate detection scan: `python3 tools/dupdef_scan.py --fail-on-issues`**

#### **Automated Duplicate Detection**

**MANDATORY:** All code changes must pass the duplicate definition scanner:

```bash
# Run before committing any code:
python3 tools/dupdef_scan.py --fail-on-issues

# For detailed report:
python3 tools/dupdef_scan.py --out duplicate_report.txt

# For JSON output:
python3 tools/dupdef_scan.py --json --out duplicate_report.json
```

**The scanner detects:**
- Duplicate class names across files
- Duplicate method names within classes
- Duplicate functions within modules
- Overload groups without implementations
- Syntax errors

**Zero tolerance policy:** Any duplicates found must be resolved before code completion.

#### **Real-World Example: PPO Cleanup (August 2025)**

**Problem:** PPO codebase had accumulated 20+ duplicate files:
```
❌ BEFORE (Confusing mess):
advanced_ppo_trainer.py
enhanced_ppo_network.py
sentio_ppo_integration.py
train_ppo_fixed.py
train_ppo_fixed_final.py
train_ppo_10_percent_monthly.py
train_ppo_optimized.py
apply_ppo_fixes_immediately.py
debug_ppo_rewards.py
test_ppo_fixes.py
... and 10+ more duplicates
```

**Solution:** Applied these rules rigorously:
```
✅ AFTER (Clean, clear):
models/ppo_trainer.py      # Single PPO training system
models/ppo_network.py      # Single neural network
models/ppo_integration.py  # Single integration module
models/ppo_trading_agent.py # Base agent system
train_ppo.py               # Single training script
```

**Result:** 
- 20+ files reduced to 5 essential files
- Zero confusion about which file to use
- All functionality preserved and improved
- Clean, maintainable codebase

**This is the standard all future development must follow.**

---

## 📁 File Management Rules

### **Documentation Cleanup**

The following files have been **REMOVED** and should **NEVER** be recreated:

#### **Removed Files**
- `INTEGRATION_COMPLETE.md`
- `ALGORITHM_COMPARISON_GUIDE.md`
- `README_ENHANCED.md`
- All files in `analysis/reports/*.md`
- All files in `docs/financeai/*.md`
- `ui/README_*.md`
- `req_requests/*.md`
- `tools/*.md`
- `entity/README.md`
- `trader-bot/README.md`
- `trader-bot/overview.md`

#### **Cleanup Commands**
```bash
# These files are removed and should not be recreated
rm -f INTEGRATION_COMPLETE.md
rm -f ALGORITHM_COMPARISON_GUIDE.md
rm -f README_ENHANCED.md
rm -rf analysis/reports/*.md
rm -rf docs/financeai/
rm -f ui/README_*.md
rm -f req_requests/*.md
rm -f tools/GUI_AND_MODEL_REQUIREMENTS.md
rm -f tools/FINANCEAI_MEGA_DOC.md
rm -f entity/README.md
rm -f trader-bot/README.md
rm -f trader-bot/overview.md
```

### **Allowed Non-Documentation Files**

These files are **PERMITTED** and serve specific functions:
- `PROJECT_RULES.md` (this file - project governance)
- `requirements*.txt` (dependency management)
- `config/*.yaml` (configuration files)
- `config/*.json` (configuration files)
- `.env.example` (environment template)

### **Git Repository Exclusions**

The following directories and files are **TEMPORARY** and must **NEVER** be committed to git:

#### **Excluded Directories**
- `bug_reports/` - Temporary bug documentation (periodically removed)
- `req_requests/` - Temporary requirement documents (periodically removed)
- `analysis/reports/` - Temporary analysis outputs
- `logs/` - Runtime logs and temporary data

#### **Excluded File Patterns**
- `*_REPORT.md` - Temporary reports
- `*_REQUIREMENTS.md` - Temporary requirements
- `*_BUG_*.md` - Bug report documents
- `*_MEGA_*.md` - Mega document outputs
- `debug_*.py` - Debug scripts (remove after use)
- `temp_*.py` - Temporary scripts (remove after use)
- `test_*.py` - Temporary test scripts (remove after use)

#### **Git Commit Rules**

**AI models MUST:**
1. **Never commit files in `bug_reports/` or `req_requests/` directories**
2. **Never commit temporary documentation files**
3. **Remove temporary files before committing**
4. **Only commit permanent code and the two allowed documentation files**

**Example of correct git workflow:**
```bash
# WRONG - includes temporary docs
git add bug_reports/ req_requests/ *.md

# RIGHT - only permanent code
git add algorithms/ models/ ui/ services/
git add docs/ARCHITECTURE.md docs/README.md  # Only these docs allowed
```

---

## 📄 Mega Document Generation Rules

**CRITICAL**: AI models must **NEVER** create mega documents manually as they consume excessive tokens and are inefficient.

### **Mandatory Usage of create_mega_document.py**

**Always use the `create_mega_document.py` tool** located in the `tools/` folder for creating comprehensive documentation:

#### **1. Bug Reports (place in `bug_reports/` folder)**

```bash
# Step 1: Create JSON configuration file
# Example: bug_reports/performance_bug_config.json
{
  "files": [
    "algorithms/problematic_strategy.py",
    "test_results.py", 
    "docs/ARCHITECTURE.md",
    "existing_analysis.md"
  ],
  "sections": [
    {
      "title": "Executive Summary",
      "content": "Brief description of the bug and impact"
    },
    {
      "title": "Root Cause Analysis",
      "content": "Detailed analysis of what's causing the issue"
    },
    {
      "title": "Proposed Solutions", 
      "content": "Implementation roadmap and fixes"
    }
  ]
}

# Step 2: Generate mega document
python3 tools/create_mega_document.py \
  --config bug_reports/performance_bug_config.json \
  --title "Performance Bug Analysis" \
  --description "Analysis of system performance issues" \
  --output bug_reports/PERFORMANCE_BUG_ANALYSIS_MEGA.md
```

#### **2. Requirement Requests (place in `req_requests/` folder)**

```bash
# Step 1: Create JSON configuration file  
# Example: req_requests/feature_enhancement_config.json
{
  "files": [
    "algorithms/current_implementation.py",
    "docs/ARCHITECTURE.md",
    "similar_feature_example.py",
    "performance_benchmarks.json"
  ],
  "sections": [
    {
      "title": "Feature Requirements",
      "content": "Detailed requirements and specifications"
    },
    {
      "title": "Success Criteria",
      "content": "How to measure successful implementation"
    },
    {
      "title": "Implementation Roadmap",
      "content": "Step-by-step implementation plan"
    }
  ]
}

# Step 2: Generate mega document
python3 tools/create_mega_document.py \
  --config req_requests/feature_enhancement_config.json \
  --title "Feature Enhancement Request" \
  --description "Request for new system capabilities" \
  --output req_requests/FEATURE_ENHANCEMENT_REQUEST_MEGA.md
```

#### **3. Performance Analysis Documents**

```bash
# For performance gap analysis, system comparisons, etc.
# Example: bug_reports/financeai_comparison_config.json
{
  "files": [
    "algorithms/ultimate_meta_strategy_selector.py",
    "algorithms/ultra_volatility_exploitation_strategy.py", 
    "test_results.json",
    "historical_performance_data.md"
  ],
  "sections": [
    {
      "title": "Performance Comparison",
      "content": "Current vs target performance metrics"
    },
    {
      "title": "Gap Analysis", 
      "content": "Identification of performance gaps and causes"
    }
  ]
}

python3 tools/create_mega_document.py \
  --config bug_reports/financeai_comparison_config.json \
  --title "FinanceAI Performance Gap Analysis" \
  --description "Comprehensive performance comparison and analysis" \
  --output bug_reports/FINANCEAI_PERFORMANCE_GAP_MEGA.md
```

### **JSON Configuration Format**

```json
{
  "files": [
    "path/to/file1.py",
    "path/to/file2.md",
    "path/to/file3.json"
  ],
  "sections": [
    {
      "title": "Section Title",
      "content": "Section content description and analysis"
    }
  ],
  "max_lines": 1000
}
```

### **Key Benefits of Using the Tool**

1. **Token Efficiency**: Avoids massive token consumption from manual document creation
2. **Standardized Format**: Consistent document structure and formatting  
3. **File Management**: Proper file inclusion and organization
4. **Metadata**: Automatic statistics and file information
5. **AI-Ready**: Optimized format for AI model analysis
6. **Size Management**: Handles large documents efficiently

### **Prohibited Actions**

❌ **NEVER DO**: Create mega documents manually in chat  
❌ **NEVER DO**: Copy/paste large amounts of source code directly  
❌ **NEVER DO**: Create comprehensive documents without using the tool  
❌ **NEVER DO**: Write documents >1KB manually in responses

✅ **ALWAYS DO**: Use `create_mega_document.py` for any document >1KB  
✅ **ALWAYS DO**: Create JSON config first, then run the tool  
✅ **ALWAYS DO**: Place outputs in appropriate folders (`bug_reports/` or `req_requests/`)  
✅ **ALWAYS DO**: Include relevant source code files in the configuration

### **Common Use Cases**

- **Bug Analysis**: Performance issues, algorithm failures, integration problems
- **Feature Requests**: New algorithm implementations, UI enhancements, system upgrades  
- **Performance Reviews**: Benchmarking, optimization analysis, comparison studies
- **Architecture Analysis**: System design reviews, component integration studies
- **Code Reviews**: Comprehensive code analysis with multiple file context

### **Tool Command Reference**

```bash
# Basic usage with config file
python3 tools/create_mega_document.py \
  --config path/to/config.json \
  --title "Document Title" \
  --description "Document description" \
  --output path/to/output.md

# Direct file list (without config)
python3 tools/create_mega_document.py \
  --files file1.py file2.md file3.json \
  --title "Document Title" \
  --description "Document description" \
  --output path/to/output.md

# With custom source directory
python3 tools/create_mega_document.py \
  --config config.json \
  --source ./algorithms \
  --title "Algorithm Analysis" \
  --description "Algorithm implementation review" \
  --output bug_reports/ALGORITHM_ANALYSIS_MEGA.md
```

---

## 🔧 Development Workflow

### **Standard Development Process**

1. **Code Changes**: Implement features following architecture rules
2. **Integration**: Ensure GUI and multi-algorithm integration
3. **Testing**: Verify functionality and performance
4. **Documentation**: Update `docs/ARCHITECTURE.md` and `docs/README.md`
5. **Cleanup**: Remove any temporary files or outdated information

### **Feature Addition Checklist**

- [ ] Integrates with multi-algorithm system
- [ ] Has GUI controls and monitoring
- [ ] Includes performance tracking
- [ ] Uses Kafka messaging appropriately
- [ ] Follows error handling patterns
- [ ] Updates `docs/ARCHITECTURE.md`
- [ ] Updates `docs/README.md`
- [ ] No new documentation files created

### **Bug Fix Checklist**

- [ ] Identifies root cause
- [ ] Implements proper fix
- [ ] Tests fix thoroughly
- [ ] Updates documentation if architecture affected
- [ ] No temporary documentation created

---

## 🎯 Quality Standards

### **Code Quality**

1. **Performance**: Sub-second response times for GUI operations
2. **Reliability**: Graceful error handling and recovery
3. **Scalability**: Support for multiple algorithms and symbols
4. **Maintainability**: Clear, documented code structure
5. **Integration**: Seamless component interaction

### **Logging Policy (Mandatory)**

All components must use centralized, structured JSON logging:

1. Initialize once via `core.json_logging.configure_json_logging()` at process start.
2. Do not use `print()` in production code. Use `logging.getLogger(__name__)` only.
3. JSON fields emitted by default: `timestamp`, `level`, `logger`, `message`, `run_id`.
4. Include when available: `algo`, `symbol`, `order_id`, `event_seq`, `event`, `component`.
5. Sinks: stdout and `logs/app.jsonl`. Errors also recorded in `logs/errors.log`.
6. Domain messages (Kafka/persistence) must carry `run_id` and `event_seq`.
7. UI and background threads must not directly mutate widgets; emit events/logs instead.

### **Documentation Quality**

1. **Accuracy**: Documentation matches current codebase exactly
2. **Completeness**: All features and components documented
3. **Clarity**: Clear instructions and explanations
4. **Currency**: No outdated information
5. **Consolidation**: All information in two files only

### **User Experience**

1. **Intuitive GUI**: Easy-to-use interface
2. **Real-time Feedback**: Live performance monitoring
3. **Professional Appearance**: Consistent theme system
4. **Reliable Operation**: Stable, predictable behavior
5. **Clear Documentation**: Easy setup and usage instructions

---

## 🚨 Enforcement

### **Automatic Checks**

AI models should verify:
- No new `.md` files created
- Both documentation files updated when needed
- Architecture changes reflected in documentation
- User-facing changes reflected in README

### **Review Requirements**

Before completing any work:
1. **Architecture Review**: Changes match documented architecture
2. **Documentation Review**: Both files are current and accurate
3. **Integration Review**: Components work together properly
4. **Performance Review**: System meets performance standards

---

## 📈 Success Metrics

### **Documentation Success**
- **Single Source of Truth**: All information in two files
- **Always Current**: Documentation matches codebase
- **User-Friendly**: Clear installation and usage instructions
- **Technically Complete**: Full architecture documentation

### **System Success**
- **Multi-Algorithm Performance**: All algorithms integrated and performing
- **GUI Functionality**: Complete control and monitoring interface
- **Real-time Operation**: Live trading and performance tracking
- **Professional Quality**: Institutional-grade trading platform

---

## 🎉 Conclusion

These rules ensure Sentio Trader maintains:
- **Clean Documentation**: Two comprehensive, current files
- **Professional Architecture**: Consistent, scalable system design
- **Quality Standards**: Reliable, high-performance operation
- **User Experience**: Clear, intuitive interface and documentation

**All AI models and contributors must follow these rules without exception.**

---

*Sentio Trader Project Rules - Ensuring Quality and Consistency*
