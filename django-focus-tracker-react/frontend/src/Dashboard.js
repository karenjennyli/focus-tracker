import React from 'react';
import {
  Box,
  Flex,
  Text,
  SimpleGrid,
  Heading,
  Container,
  VStack,
  Spacer,
  Button,
  useColorModeValue,
} from '@chakra-ui/react';

const Dashboard = () => {
  const cardBg = useColorModeValue('gray.100', 'gray.700');

  const Card = ({ title, content }) => (
    <Box bg={cardBg} p={5} shadow="md" borderRadius="md">
      <Heading fontSize="xl">{title}</Heading>
      <Text mt={4}>{content}</Text>
    </Box>
  );

  return (
    <Box>
      <Flex as="nav" p={4} bg="blue.500" color="white" align="center">
        <Heading size="md">Dashboard</Heading>
        <Spacer />
        <Button colorScheme="blue" variant="outline">Logout</Button>
      </Flex>

      <Container maxW="container.xl" mt={5}>
        <VStack spacing={4} align="stretch">
          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={5}>
            <Card title="Card 1" content="This is the first card's content." />
            <Card title="Card 2" content="This is the second card's content." />
            <Card title="Card 3" content="This is the third card's content." />
          </SimpleGrid>
        </VStack>
      </Container>
    </Box>
  );
};

export default Dashboard;
