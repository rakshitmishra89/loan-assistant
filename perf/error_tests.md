# Error Test Cases
## 1. Missing Input
Input: {}
Expected: "Missing required fields"

## 2. Invalid Salary
Input: salary = -1000
Expected: "Invalid salary"

## 3. Invalid Loan Amount
Input: loan = "abc"
Expected: "Invalid input"

## 4. Large Input
Input: very long text
Expected: No crash

## 5. Unsafe Input
Input: abusive message
Expected: Blocked

## 6. API Down
Expected: Proper error message

## 7. Unsupported File
Expected: "File not supported"