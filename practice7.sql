create database college;
use college;
CREATE TABLE Departments (
    DepartmentID INT PRIMARY KEY,
    DepartmentName VARCHAR(50),
    HOD VARCHAR(50),
    Building VARCHAR(30)
);

INSERT INTO Departments VALUES
(1, 'Computer Science', 'Dr. Mehta', 'A Block'),
(2, 'Mechanical', 'Dr. Sharma', 'B Block'),
(3, 'Civil', 'Dr. Singh', 'C Block'),
(4, 'Electronics', 'Dr. Gupta', 'D Block'),
(5, 'Electrical', 'Dr. Verma', 'E Block'),
(6, 'MBA', 'Dr. Khan', 'F Block'),
(7, 'Pharmacy', 'Dr. Das', 'G Block'),
(8, 'Biotechnology', 'Dr. Roy', 'H Block'),
(9, 'Mathematics', 'Dr. Jain', 'I Block'),
(10, 'Physics', 'Dr. Patel', 'J Block');

CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    StudentName VARCHAR(50),
    Gender VARCHAR(10),
    Age INT,
    DepartmentID INT,
    Semester INT,
    City VARCHAR(30),
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

INSERT INTO Students VALUES
(101, 'Amit Sharma', 'Male', 19, 1, 2, 'Delhi'),
(102, 'Priya Singh', 'Female', 20, 2, 4, 'Lucknow'),
(103, 'Rahul Verma', 'Male', 21, 3, 6, 'Jaipur'),
(104, 'Neha Gupta', 'Female', 18, 1, 2, 'Noida'),
(105, 'Arjun Kumar', 'Male', 20, 4, 4, 'Kanpur'),
(106, 'Sneha Patel', 'Female', 19, 2, 2, 'Bhopal'),
(107, 'Mohit Yadav', 'Male', 22, 3, 8, 'Agra'),
(108, 'Pooja Mishra', 'Female', 21, 4, 6, 'Meerut'),
(109, 'Rohan Das', 'Male', 20, 2, 4, 'Patna'),
(110, 'Anjali Jain', 'Female', 22, 1, 8, 'Indore');

CREATE TABLE Courses (
    CourseID VARCHAR(10) PRIMARY KEY,
    CourseName VARCHAR(50),
    DepartmentID INT,
    Credits INT,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

INSERT INTO Courses VALUES
('C101', 'DBMS', 1, 4),
('C102', 'Data Structures', 1, 4),
('C103', 'Thermodynamics', 2, 3),
('C104', 'Fluid Mechanics', 3, 4),
('C105', 'Digital Electronics', 4, 4),
('C106', 'Power Systems', 5, 3),
('C107', 'Marketing', 6, 3),
('C108', 'Pharmacology', 7, 4),
('C109', 'Linear Algebra', 9, 3),
('C110', 'Quantum Physics', 10, 4);

CREATE TABLE Marks (
    MarkID INT PRIMARY KEY,
    StudentID INT,
    CourseID VARCHAR(10),
    Marks INT,
    Grade VARCHAR(5),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);

INSERT INTO Marks VALUES
(1, 101, 'C101', 88, 'A'),
(2, 102, 'C103', 76, 'B'),
(3, 103, 'C104', 91, 'A+'),
(4, 104, 'C102', 84, 'A'),
(5, 105, 'C105', 69, 'C'),
(6, 106, 'C103', 80, 'A'),
(7, 107, 'C104', 74, 'B'),
(8, 108, 'C105', 95, 'A+'),
(9, 109, 'C103', 66, 'C'),
(10, 110, 'C101', 90, 'A+');

/* Questions Answers */
select * from students;
select * from departments;
select studentname, city from students;
select * from students where city like '%a%';
select * from courses where credits = 4;
select * from departments order by departmentname asc;
select * from students order by age desc;
select * from students order by age limit 5;
select count(*) as total_Students from students;
select avg(age) from students;
select avg(marks) from marks;
select departmentname , count(*) as total_students from departments group by departmentname;
select city , count(*) as total_students from students group by city;
select d.departmentname , avg(s.age) as avg_age from departments d join students s on 
d.departmentid = s.departmentid group by departmentname;
select c.coursename , max(m.marks) as highest_marks from courses c join marks m on 
c.courseid = m.courseid group by coursename;
select count(*) as fstudents from students where gender='female';
select d.departmentname , count(*) as total_courses from departments d join courses c on d.departmentid = c.departmentid group by departmentname;
select d.departmentname , count(s.studentid) > 1  from departments d join students s on d.departmentid = s.departmentid 
group by departmentname;
select d.departmentname , avg(s.age) as avg_age from departments d join students s
 on d.departmentid = s.departmentid group by departmentname;
select d.departmentname , count(credits) as total_credits from departments d join courses c
on d.departmentid = c.departmentid group by departmentname;
select avg(credits) as avg_credits from courses;
select s.studentname , sum(m.marks) as total_marks from students s join marks m 
on s.studentid = m.studentid group by studentname;
select s.studentname , d.departmentname from students s join departments d on  d.departmentid = s.departmentid ;
select s.studentname , d.Hod from students s join departments d on  d.departmentid = s.departmentid ;
select c.coursename , d.departmentname from courses c join departments d on  d.departmentid = c.departmentid ;
select s.studentname , c.coursename , m.marks from students s inner join marks m on s.studentid = m.studentid 
join courses c on m.courseid = c.courseid;
select s.studentname , d.departmentname from students s join departments d
 on d.departmentid = s.departmentid where departmentname ='computer science' ;
 








