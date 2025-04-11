# ML Interview Quiz - Product Requirement Document

## 1. Product Overview
ML Interview Quiz is an interactive web application designed to help users prepare for machine learning interviews through a quiz-based learning approach. The application provides immediate feedback, tracks user performance, and offers additional learning resources.

## 2. Core Features

### 2.1 Quiz System
- **Random Question Selection**
  - Randomly selects 15 questions from a pool of 70 machine learning questions
  - Questions cover various ML topics including algorithms, concepts, and best practices
  - Questions are shuffled for each attempt

### 2.2 Question Interface
- **Question Display**
  - Shows current question number and progress bar
  - Displays multiple choice options (4 options per question)
  - Options are randomly ordered for each question

### 2.3 Answer Feedback System
- **Immediate Feedback**
  - Shows correct/incorrect status immediately after answer selection
  - Highlights selected answer in red (incorrect) or green (correct)
  - Displays explanation for the answer
  - Provides "Learn more" link to external resources
  - Manual progression to next question via "Next Question" button

### 2.4 Progress Tracking
- **Score History**
  - Records and displays scores from previous attempts
  - Shows number of correct answers per attempt
  - Persists data using local storage

### 2.5 Learning Analytics
- **Frequently Missed Problems**
  - Tracks questions that users answer incorrectly
  - Shows top 5 most frequently missed questions
  - Displays count of times each question was missed
  - Provides explanations and reference links for improvement

## 3. User Interface

### Navigation Structure
- Main Page
  - Quiz Tab
    - Question View
    - Feedback View
    - Results View
  - Score History Tab
  - Missed Problems Tab

### Key User Flows
- Quiz Flow: Question → Feedback → Next Question → Results
- Review Flow: History View ↔ Quiz View ↔ Missed Problems View

## 4. Technical Requirements

### 4.1 Frontend
- Next.js 13+ framework
- React with TypeScript
- Tailwind CSS for styling
- Client-side state management
- Responsive design for all screen sizes

### 4.2 Data Management
- Local storage for persisting:
  - Score history
  - Frequently missed questions
  - User progress

### 4.3 Performance
- Client-side rendering for quiz interface
- Optimized for quick feedback
- Smooth transitions between questions

## 5. User Flow

1. **Starting the Quiz**
   - User lands on quiz page
   - System randomly selects 15 questions
   - Progress bar shows 0/15 completion

2. **During the Quiz**
   - User selects an answer
   - System provides immediate feedback
   - User reviews explanation and resources
   - User manually proceeds to next question

3. **Completing the Quiz**
   - System displays final score
   - Shows review of all questions
   - Offers options to:
     - Try again
     - View history
     - Review missed problems

## 6. UI Components

### 6.1 Navigation
- Three tabs:
  - Quiz
  - Score History
  - Missed Problems

### 6.2 Quiz Interface Components
- Question counter
- Progress bar
- Question text
- Answer options (4)
- Feedback section
  - Correct/Incorrect indicator
  - Explanation text
  - Reference link
  - Next question button

### 6.3 Results View Components
- Final score display
- Question review list
- Navigation buttons
- Performance summary

## 7. Future Enhancements
1. User accounts for progress tracking
2. Category-based question filtering
3. Difficulty levels
4. Social sharing features
5. Performance analytics
6. Custom quiz creation
7. Timed quiz mode

## 8. Success Metrics
1. User engagement (attempts per session)
2. Learning effectiveness (score improvement over time)
3. Resource utilization (reference link clicks)
4. User retention (return visits)

## 9. Development Notes

### Current Implementation
- Built with Next.js and TypeScript
- Uses Tailwind CSS for styling
- Implements client-side storage
- Deployed on GitHub Pages

### Known Issues
- TypeScript linting warnings in tab navigation
- Need to handle hydration issues carefully
- Local storage limitations

### Maintenance
- Regular updates to question pool
- Periodic review of reference links
- Performance monitoring

---

*Last Updated: [Current Date]*
*Version: 1.0.0*
*Author: [Your Team]*

Feel free to modify this document as needed for your project requirements. 